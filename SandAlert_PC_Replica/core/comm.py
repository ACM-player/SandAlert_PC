# =============================================================================
# SandAlert PC - 通信引擎
# =============================================================================
# 说明：本文件负责数据采集与通信。
#
# 包含两个线程类：
#   1. SerialReaderThread  — 从真实串口（或虚拟串口）读取数据
#   2. FileReplayThread    — 从本地 .txt 文件回放数据（无硬件时的替代方案）
#
# 两个线程的 data_received 信号格式完全一致（发射 numpy 数组），
# 因此 UI 层无需关心数据来源，只需绑定 update_from_serial 槽函数即可。
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
