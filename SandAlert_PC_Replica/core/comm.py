# =============================================================================
# SandAlert PC - 通信引擎
# =============================================================================
# 说明：本文件负责数据采集与通信。
#
# 包含三个线程类：
#   1. SerialReaderThread     — 从真实串口（或虚拟串口）读取声学波形数据
#   2. FileReplayThread       — 从本地 .txt 文件回放数据（无硬件时的替代方案）
#   3. ModbusReaderThread     — 通过 Modbus RTU/TCP 协议读取 SandAlert 仪表全量参数
#
# 前两个线程发射 numpy 数组（波形数据），第三个线程发射 dict（解析后的工业参数）。
# UI 层的 update_from_serial 通过 isinstance 判断数据类型并分别处理。
# =============================================================================

import numpy as np
from PySide6.QtCore import QThread, Signal
import time
import glob
import os


# =============================================================================
# SerialReaderThread：真实/虚拟串口数据读取线程
# =============================================================================
class SerialReaderThread(QThread):
    """
    串口数据读取线程：从 CH340（或虚拟串口）读取高频声学数据。
    
    支持两种数据格式（自动检测）：
    
    格式 1 — 逗号分隔的 4 通道（推荐，兼容未来硬件）:
        -2.835,-2.840,-2.838,-2.842
        -2.841,-2.843,-2.839,-2.844
    
    格式 2 — 单值（向后兼容）:
        -2.835
        -2.841
    
    发射的数据格式：
    - 如果检测到 4 通道数据：发射 (4, chunk_size) ndarray
    - 如果检测到单值数据：发射 (chunk_size,) ndarray（向后兼容）
    
    每凑满 chunk_size（默认 50）帧打包发射一次。
    注：在 4 通道模式下，50 帧 = 50×4 = 200 个 float
    """
    # 修复点：必须用 object 类型，否则 PySide6 会在跨线程传递 numpy 数组时静默丢弃数据
    data_received = Signal(object)
    status_msg = Signal(str)

    def __init__(self, port, baudrate, parent=None):
        """
        初始化串口线程
        
        参数:
            port: 串口名称，如 "/dev/tty.usbserial-1410" 或 "/dev/ttys001"
            baudrate: 波特率，如 19200、115200
        """
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self._is_running = False
        
        # 数据缓冲区
        # - 单通道模式：buffer 为 list[float]，每攒 50 个发射
        # - 4 通道模式：buffer 为 list[tuple]，每攒 50 帧发射
        self.chunk_size = 50
        self.buffer = []
        
        # 数据格式检测标志
        # None = 未检测, 1 = 单值模式, 4 = 4 通道模式
        self._format = None

    def _detect_format(self, decoded: str) -> int:
        """
        检测首次有效数据的格式：单值还是逗号分隔的 4 通道。
        
        参数
        ----------
        decoded : str
            解码后的一行原始数据。
        
        返回
        -------
        int
            1（单值模式）或 4（4 通道模式）。
        """
        # 按逗号分割
        parts = decoded.strip().split(',')
        
        if len(parts) == 4:
            return 4
        else:
            return 1

    def _parse_line(self, decoded: str):
        """
        解析一行数据，根据检测到的格式存入缓冲区。
        
        参数
        ----------
        decoded : str
            解码后的一行原始数据。
        """
        if self._format == 4:
            # 4 通道模式：按逗号分割，解析 4 个 float
            parts = decoded.strip().split(',')
            if len(parts) != 4:
                return  # 格式异常，静默丢弃
            try:
                ch1, ch2, ch3, ch4 = map(float, parts)
                self.buffer.append((ch1, ch2, ch3, ch4))
            except ValueError:
                pass  # 脏数据，丢弃
        else:
            # 单通道模式：直接解析为 float
            try:
                val = float(decoded)
                self.buffer.append(val)
            except ValueError:
                pass  # 脏数据，丢弃

    def _flush_buffer(self):
        """
        将缓冲区打包为 numpy 数组并发射。
        
        4 通道模式：发射 shape (4, chunk_size) 数组
        单通道模式：发射 shape (chunk_size,) 数组（向后兼容）
        """
        if not self.buffer:
            return
        
        if self._format == 4:
            # 4 通道模式：转置为 (4, N)
            np_data = np.array(self.buffer, dtype=np.float32).T  # shape (4, N)
        else:
            # 单通道模式：简单列表转数组
            np_data = np.array(self.buffer, dtype=np.float32)
        
        self.data_received.emit(np_data)
        self.buffer.clear()

    def run(self):
        """
        线程的核心运行逻辑。
        
        流程：
        1. 尝试打开串口
        2. 循环读取每行数据
        3. 自动检测数据格式（单值 vs. 4 通道）
        4. 解析并存入缓冲区
        5. 缓冲区满时打包发射
        """
        import serial  # 延迟导入，避免没有安装 pyserial 时导入报错
        
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=int(self.baudrate),
                timeout=1.0  # 读取超时 1 秒
            )
            self._is_running = True
            self.status_msg.emit(f"✅ 已连接到 {self.port}，波特率 {self.baudrate} bps")
            
        except serial.SerialException as e:
            self.status_msg.emit(f"❌ 连接失败: {str(e)}")
            return
        except Exception as e:
            self.status_msg.emit(f"❌ 未知错误: {str(e)}")
            return

        # 主读取循环
        format_reported = False
        while self._is_running and self.serial_port.is_open:
            try:
                # 按行读取
                raw_line = self.serial_port.readline()
                
                if not raw_line:
                    # 超时未读到数据，继续循环
                    continue
                
                # 解码并去除首尾空白字符
                decoded = raw_line.decode('utf-8', errors='ignore').strip()
                
                if not decoded:
                    continue
                
                # ---- 自动检测数据格式（仅在首次有效数据时执行） ----
                if self._format is None:
                    self._format = self._detect_format(decoded)
                    if self._format == 4:
                        self.status_msg.emit(
                            "✅ 检测到 4 通道阵列数据格式 (逗号分隔)"
                        )
                    else:
                        self.status_msg.emit(
                            "✅ 检测到单通道数据格式 (向后兼容模式)"
                        )
                
                # ---- 解析当前行 ----
                self._parse_line(decoded)
                
                # ---- 缓冲区满则发射 ----
                if len(self.buffer) >= self.chunk_size:
                    self._flush_buffer()
                        
            except serial.SerialException as e:
                self.status_msg.emit(f"⚠️ 串口读取错误: {str(e)}")
                break
            except Exception as e:
                self.status_msg.emit(f"⚠️ 读取异常: {str(e)}")
                break
        
        # 线程退出前，清空剩余数据
        if self.buffer:
            self._flush_buffer()

    def stop(self):
        """
        安全停止线程并关闭串口。
        由主窗口的 closeEvent 或切换串口时调用。
        """
        self._is_running = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
                self.status_msg.emit("🔌 串口已关闭")
            except Exception:
                pass
        self.wait(2000)  # 最多等待 2 秒


# =============================================================================
# FileReplayThread：文件数据回放线程（无硬件时的替代方案）
# =============================================================================
class FileReplayThread(QThread):
    """
    批量数据回放线程：遍历指定文件夹下的所有 txt 文件，按顺序无缝回放。
    
    当没有真实硬件或虚拟串口时，可以使用此线程从本地文件回放数据，
    用于调试和演示。
    """
    data_received = Signal(object)
    status_msg = Signal(str)

    def __init__(self, data_dir, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir 
        self._is_running = False
        self.chunk_size = 50 
        self.delay_ms = 0.05 

    def run(self):
        self._is_running = True
        
        search_pattern = os.path.join(self.data_dir, "*.txt")
        txt_files = sorted(glob.glob(search_pattern))
        
        if not txt_files:
            self.status_msg.emit(f"❌ 文件夹为空或未找到 .txt 文件: {self.data_dir}")
            return

        self.status_msg.emit(f"📂 找到 {len(txt_files)} 个文件，开始顺序回放...")

        for file_path in txt_files:
            if not self._is_running:
                break
                
            file_name = os.path.basename(file_path)
            self.status_msg.emit(f"📄 正在回放: {file_name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                    tokens = text_content.split()
                    
                    buffer = []
                    for token in tokens:
                        if not self._is_running:
                            break 
                            
                        try:
                            buffer.append(float(token))
                        except ValueError:
                            continue
                            
                        if len(buffer) >= self.chunk_size:
                            np_data = np.array(buffer, dtype=np.float32)
                            self.data_received.emit(np_data)
                            buffer = [] 
                            time.sleep(self.delay_ms) 
                            
            except Exception as e:
                self.status_msg.emit(f"⚠️ 读取 {file_name} 出错: {str(e)}")
                continue 
                
        if self._is_running:
            self.status_msg.emit("✅ 回放完成，所有文件已处理。")

    def stop(self):
        self._is_running = False
        self.wait()


# =============================================================================
# ModbusReaderThread：Modbus RTU/TCP 数据读取线程
# =============================================================================
class ModbusReaderThread(QThread):
    """
    Modbus 数据读取线程：通过 pymodbus 协议批量读取 SandAlert 仪表全量参数。
    
    支持两种模式：
      - RTU 模式：通过串口连接真实物理仪表（现场部署）
      - TCP 模式：连接本地 Mock 服务器（测试联调）
    
    通信参数：
      - 波特率: 19200
      - 字节大小: 8，停止位: 1，校验位: 'E' (偶校验)
      - 从机地址 (Slave ID): 126
    
    读取逻辑：
      - 发送 client.read_input_registers(address=0, count=11, device_id=126)
      - 从 result.registers（长度 11）中按索引解包
      - 通过 data_received 发射 dict 给 UI 线程
    
    dict 格式:
        {
            'ips': int,            # 瞬时砂击率
            'mass_rate': float,    # 平均质量/秒 (g/s)
            'total_mass': float,   # 累计总砂量 (g)
            'relay_status': int,   # 警报位映射
            'current_ma': float,   # 模拟电流 (mA)
            'raw_registers': list  # 原始 11 个寄存器值 (调试用)
        }
    """
    data_received = Signal(object)  # 发射 dict，而非 numpy 数组
    status_msg = Signal(str)

    # ---- Modbus RTU 串口参数 ----
    MODBUS_BAUDRATE = 19200
    MODBUS_BYTESIZE = 8
    MODBUS_STOPBITS = 1
    MODBUS_PARITY = 'E'       # 偶校验 (Even)
    MODBUS_SLAVE_ID = 126

    # ---- 读取参数 ----
    READ_ADDRESS = 0
    READ_COUNT = 11
    POLL_INTERVAL = 0.5  # 轮询间隔（秒）

    def __init__(self, port_or_host, use_tcp=False, tcp_port=5020, parent=None):
        """
        初始化 Modbus 读取线程。
        
        参数:
            port_or_host: RTU 模式=串口路径(e.g. /dev/tty.usbserial-1410)；
                          TCP 模式=主机地址(e.g. 127.0.0.1)
            use_tcp: True=TCP 模式（本地测试）, False=RTU 模式（真实硬件）
            tcp_port: TCP 端口号（仅 TCP 模式），默认 5020
            parent: QObject 父对象
        """
        super().__init__(parent)
        self.port_or_host = port_or_host
        self.use_tcp = use_tcp
        self.tcp_port = tcp_port
        self.client = None
        self._is_running = False

    def run(self):
        """
        线程核心运行逻辑。
        
        流程：
        1. 根据模式创建 ModbusSerialClient（RTU 或 TCP）
        2. 尝试连接
        3. 循环轮询：批量读 Input Registers → 解包 → emit
        4. 所有异常都被 try...except 捕获，绝不崩溃 UI
        """
        try:
            if self.use_tcp:
                # ====== TCP 模式（本地调试） ======
                # ModbusTcpClient：纯 TCP 连接，无需任何串口参数
                from pymodbus.client import ModbusTcpClient

                self.client = ModbusTcpClient(
                    host=self.port_or_host,
                    port=self.tcp_port,
                    timeout=3.0,
                )

                self.status_msg.emit(
                    f"🔌 Modbus TCP 连接中: {self.port_or_host}:{self.tcp_port} ..."
                )
            else:
                # ====== RTU 模式（真实仪表） ======
                from pymodbus.client import ModbusSerialClient

                self.client = ModbusSerialClient(
                    port=self.port_or_host,
                    baudrate=self.MODBUS_BAUDRATE,
                    bytesize=self.MODBUS_BYTESIZE,
                    stopbits=self.MODBUS_STOPBITS,
                    parity=self.MODBUS_PARITY,
                    timeout=2.0,
                )
                self.status_msg.emit(
                    f"🔌 Modbus RTU 连接中: {self.port_or_host} @ {self.MODBUS_BAUDRATE} bps ..."
                )


            # ---- 尝试连接 ----
            connected = self.client.connect()
            if not connected:
                self.status_msg.emit(
                    f"❌ Modbus 连接失败: {self.port_or_host}"
                )
                return

            self._is_running = True
            self.status_msg.emit(
                f"✅ Modbus {'TCP' if self.use_tcp else 'RTU'} 已连接 | "
                f"Slave ID={self.MODBUS_SLAVE_ID} | 轮询间隔={self.POLL_INTERVAL}s"
            )

        except Exception as e:
            self.status_msg.emit(f"❌ Modbus 初始化错误: {str(e)}")
            return

        # ========== 主轮询循环 ==========
        consecutive_errors = 0
        max_consecutive_errors = 5  # 连续 5 次失败后自动断开

        while self._is_running:
            try:
                # ---- 批量连读 11 个 Input Registers ----
                # 地址从 0 开始，读取 11 个，从机 ID=126
                result = self.client.read_input_registers(
                    address=self.READ_ADDRESS,
                    count=self.READ_COUNT,
                    device_id=self.MODBUS_SLAVE_ID,
                )

                # ---- 检查读取是否成功 ----
                if result is None or hasattr(result, 'isError') and result.isError():
                    consecutive_errors += 1
                    self.status_msg.emit(
                        f"⚠️ Modbus 读取失败 (第 {consecutive_errors} 次) | "
                        f"错误: {result}"
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        self.status_msg.emit(
                            "❌ Modbus 连续多次读取失败，自动断开"
                        )
                        break
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- 读取成功，重置错误计数 ----
                consecutive_errors = 0
                registers = result.registers

                # ====== 【诊断日志】打印原始寄存器值 ======
                print(f"[MODBUS] 📥 原始寄存器值 ({len(registers)} 个): {registers}")

                if len(registers) < self.READ_COUNT:
                    print(f"[MODBUS] ❌ 数据不完整: 期望 {self.READ_COUNT} 个, 实际 {len(registers)} 个")
                    self.status_msg.emit(
                        f"⚠️ Modbus 返回数据不完整: "
                        f"期望 {self.READ_COUNT} 个，实际 {len(registers)} 个"
                    )
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ================================================================
                #  严格按照索引解包（对应任务文档要求）
                # ================================================================
                ips = registers[0]                  # 30001 瞬时砂击率
                mass_rate_raw = registers[2]        # 30003 平均质量/秒（×100存储）
                total_mass_top = registers[4]       # 30005 累计总砂量高位
                total_mass_bottom = registers[5]    # 30006 累计总砂量低位
                relay_status = registers[8]         # 30009 警报位映射
                current_raw = registers[9]          # 30010 模拟电流（×100存储）

                # ====== 【诊断日志】打印解包后的中间值 ======
                print(f"[MODBUS] 📊 解包: ips={ips}  mass_rate_raw={mass_rate_raw}  "
                      f"top={total_mass_top}  bottom={total_mass_bottom}  "
                      f"relay=0x{relay_status:02X}  current_raw={current_raw}")


                # ---- 解码真实物理值 ----
                # mass_rate: ×100 存储，转回 g/s
                mass_rate = mass_rate_raw / 100.0
                # total_mass: 高低位拼接 (top << 16 | bottom)
                total_mass = (total_mass_top << 16) + total_mass_bottom
                # current: ×100 存储，转回 mA
                current_ma = current_raw * 0.01

                # ---- 打包为 dict 发射 ----
                modbus_data = {
                    'ips': ips,
                    'mass_rate': mass_rate,
                    'total_mass': float(total_mass),
                    'relay_status': relay_status,
                    'current_ma': current_ma,
                    'raw_registers': list(registers),  # 完整原始数据供调试
                }

                self.data_received.emit(modbus_data)

                # ---- 等待下一次轮询 ----
                time.sleep(self.POLL_INTERVAL)

            except Exception as e:
                # ====== 【执行纪律】任何异常绝不传播到 UI 线程 ======
                consecutive_errors += 1
                self.status_msg.emit(
                    f"⚠️ Modbus 轮询异常: {str(e)} (第 {consecutive_errors} 次)"
                )
                if consecutive_errors >= max_consecutive_errors:
                    self.status_msg.emit(
                        "❌ Modbus 连续异常，自动断开连接"
                    )
                    break
                time.sleep(self.POLL_INTERVAL)

        # ---- 线程结束，清理连接 ----
        self._cleanup()

    def _cleanup(self):
        """安全关闭 Modbus 客户端连接"""
        try:
            if self.client and hasattr(self.client, 'close'):
                self.client.close()
                self.status_msg.emit("🔌 Modbus 连接已关闭")
        except Exception:
            pass

    def stop(self):
        """安全停止 Modbus 轮询线程"""
        self._is_running = False
        self._cleanup()
        self.wait(2000)


# =============================================================================
# EdgeGatewayThread：连接本机独立 Edge 采集进程
# =============================================================================
class EdgeGatewayThread(QThread):
    """接收 Edge 本地网关发布的原始波形与服务器分析结果。"""

    waveform_received = Signal(object)
    event_received = Signal(object)
    status_msg = Signal(str)

    def __init__(self, url="ws://127.0.0.1:8766", parent=None):
        super().__init__(parent)
        self.url = url
        self._is_running = False
        self._socket = None

    def run(self):
        import json
        import websocket
        from shared.waveform_codec import decode_waveform

        self._is_running = True
        try:
            self._socket = websocket.create_connection(self.url, timeout=2.0)
            self._socket.settimeout(1.0)
            self.status_msg.emit(f"✅ Edge Gateway 已连接: {self.url}")
            while self._is_running:
                try:
                    message = self._socket.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(message, bytes):
                    _, array = decode_waveform(message)
                    self.waveform_received.emit(array)
                else:
                    payload = json.loads(message)
                    self.event_received.emit(payload)
        except Exception as exc:
            if self._is_running:
                self.status_msg.emit(f"❌ Edge Gateway 连接异常: {exc}")
        finally:
            self._close_socket()

    def _close_socket(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def stop(self):
        self._is_running = False
        self._close_socket()
        self.wait(2000)
