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
    
    数据格式：每行一个浮点数（电压值），以换行符分隔。
    例如：
        -2.835
        -2.841
        -2.798
    
    每凑满 chunk_size（默认 50）个数据点，打包成 numpy 数组发射一次。
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
        
        # 数据缓冲区：每积攒 50 个数据点发射一次，避免 UI 刷新过快卡死
        self.chunk_size = 50
        self.buffer = []

    def run(self):
        """
        线程的核心运行逻辑。
        
        流程：
        1. 尝试打开串口
        2. 循环读取每行数据
        3. 解析为浮点数，存入缓冲区
        4. 缓冲区满时打包发射
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
        while self._is_running and self.serial_port.is_open:
            try:
                # 按行读取（每行一个浮点数）
                raw_line = self.serial_port.readline()
                
                if not raw_line:
                    # 超时未读到数据，继续循环
                    continue
                
                # 解码并去除首尾空白字符
                decoded = raw_line.decode('utf-8', errors='ignore').strip()
                
                if decoded:
                    try:
                        voltage_val = float(decoded)
                        self.buffer.append(voltage_val)
                        
                        # 缓冲区满了，打包发射
                        if len(self.buffer) >= self.chunk_size:
                            np_data = np.array(self.buffer, dtype=np.float32)
                            self.data_received.emit(np_data)
                            self.buffer.clear()
                            
                    except ValueError:
                        # 忽略无法转换为 float 的脏数据（如串口刚连接时的乱码）
                        pass
                        
            except serial.SerialException as e:
                self.status_msg.emit(f"⚠️ 串口读取错误: {str(e)}")
                break
            except Exception as e:
                self.status_msg.emit(f"⚠️ 读取异常: {str(e)}")
                break

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
