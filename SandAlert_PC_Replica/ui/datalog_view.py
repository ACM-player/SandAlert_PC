# import numpy as np
# import pyqtgraph as pg
# from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
# from PySide6.QtCore import Qt

# class DataLogWidget(QWidget):
#     """
#     DataLogging 模式面板：用于查看和分析历史出砂趋势
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.layout = QVBoxLayout(self)
        
#         # 1. 顶部控制栏
#         self.control_layout = QHBoxLayout()
#         self.btn_load_log = QPushButton("Load Historical Log Data...")
#         self.btn_load_log.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 5px;")
        
#         self.lbl_info = QLabel("Status: Waiting for data file...")
#         self.lbl_info.setStyleSheet("color: #555; font-style: italic;")
        
#         self.control_layout.addWidget(self.btn_load_log)
#         self.control_layout.addWidget(self.lbl_info)
#         self.control_layout.addStretch()
#         self.layout.addLayout(self.control_layout)
        
#         # 2. PyQtGraph 宏观趋势图
#         pg.setConfigOptions(antialias=True)
#         self.plot_widget = pg.PlotWidget(title="Historical Sand Production Trend")
#         self.plot_widget.setBackground('w')
#         self.plot_widget.setLabel('left', 'Cumulative Mass', units='g')
#         self.plot_widget.setLabel('bottom', 'Time (Samples / Hours)')
#         self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
#         self.layout.addWidget(self.plot_widget)
        
#         # 使用带半透明填充的曲线，显得非常具有商业软件的高级感
#         self.trend_curve = self.plot_widget.plot(
#             pen=pg.mkPen('#ff7f0e', width=2), 
#             fillLevel=0, 
#             brush=(255, 127, 14, 50) # 橙色半透明填充
#         )
        
#         # 绑定按钮事件
#         self.btn_load_log.clicked.connect(self.simulate_loading_log)

#     def simulate_loading_log(self):
#         """
#         这里暂时模拟读取原版硬件的内部存储日志。
#         在真实场景中，你会用 QFileDialog 让用户选中导出的 CSV 或 DAT 历史文件。
#         """
#         # 弹窗假装选文件
#         file_path, _ = QFileDialog.getOpenFileName(self, "选择历史日志文件", "", "Log Files (*.log *.csv *.txt)")
        
#         if file_path:
#             # 模拟生成长达一个月的宏观出砂数据：大部分时间不出砂，偶尔一阵集中出砂
#             data_points = 5000 
#             bursts = np.random.choice([0, 0, 0, 0, 0, 0.2, 1.5, 0], size=data_points)
            
#             # 计算随时间累加的总出砂量
#             cumulative_mass = np.cumsum(bursts)
            
#             # 渲染到图表
#             self.trend_curve.setData(cumulative_mass)
            
#             total_sand = cumulative_mass[-1]
#             self.lbl_info.setText(f"Loaded: {file_path.split('/')[-1]} | Total Sand Produced: {total_sand:.2f} g")
#             self.lbl_info.setStyleSheet("color: green; font-weight: bold;")



######### version2
import os
import glob
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from core.dsp_engine import DSPEngine

class LogParserThread(QThread):
    """
    历史日志高速解析线程：在后台遍历所有文件，利用 DSP 引擎提取能量并计算累积出砂量
    """
    progress_update = Signal(int, int, str) # 传出：当前进度、总数、文件名
    data_ready = Signal(np.ndarray, float)  # 传出：累加数组、最终总质量
    error_msg = Signal(str)

    def __init__(self, data_dir, threshold, factor):
        super().__init__()
        self.data_dir = data_dir
        self.threshold = threshold
        self.factor = factor
        # 独立实例化一个 DSP 引擎，专用于历史数据分析
        self.dsp = DSPEngine(sample_rate=1000)
        # ====== 安全退出标志位：用于 closeEvent 优雅关闭线程 ======
        self._is_running = False

    def run(self):
        self._is_running = True
        search_pattern = os.path.join(self.data_dir, "*.txt")
        txt_files = sorted(glob.glob(search_pattern))

        if not txt_files:
            self.error_msg.emit("No .txt files found in the selected folder.")
            return

        cumulative_mass_list = [0.0]
        total_mass = 0.0
        total_files = len(txt_files)

        for i, file_path in enumerate(txt_files):
            # ====== 检查是否被要求停止（用户关闭窗口时触发） ======
            if not self._is_running:
                break

            file_name = os.path.basename(file_path)
            self.progress_update.emit(i + 1, total_files, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    tokens = f.read().split()
                    buffer = []
                    for token in tokens:
                        if not self._is_running:
                            break
                        try:
                            buffer.append(float(token))
                        except ValueError:
                            pass
                    
                    if buffer:
                        raw_data = np.array(buffer, dtype=np.float32)
                        # ====== 核心计算 ======
                        # 将整个文件的数据送入 DSP 提取纯声学能量
                        _, _, _, _, window_energy, _ = self.dsp.process_window(raw_data, self.threshold, self.factor)
                        
                        # 换算成物理克数并累加
                        file_mass = window_energy / self.factor
                        total_mass += file_mass
                        cumulative_mass_list.append(total_mass)
            except Exception as e:
                pass
        
        # 解析完毕，将完整的累加数组返回给 UI 画图
        self.data_ready.emit(np.array(cumulative_mass_list), total_mass)

    # ====== 新增：安全停止方法（供 closeEvent 调用） ======
    def stop(self):
        """设置停止标志位，让 run() 中的循环优雅退出"""
        self._is_running = False


class DataLogWidget(QWidget):
    """
    DataLogging 模式面板：真实解析历史数据并绘制宏观趋势
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 1. 顶部控制栏
        self.control_layout = QHBoxLayout()
        self.btn_load_log = QPushButton("Load Historical Log Data...")
        self.btn_load_log.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 5px;")
        
        # 加入进度条组件
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)
        
        self.lbl_info = QLabel("Status: Ready. Please load a folder.")
        self.lbl_info.setStyleSheet("color: #555; font-style: italic;")
        
        self.control_layout.addWidget(self.btn_load_log)
        self.control_layout.addWidget(self.progress_bar)
        self.control_layout.addWidget(self.lbl_info)
        self.control_layout.addStretch()
        self.layout.addLayout(self.control_layout)
        
        # 2. PyQtGraph 宏观趋势图
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget(title="Historical Sand Production Trend")
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Cumulative Mass', units='g')
        self.plot_widget.setLabel('bottom', 'Time (File Segments)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)
        
        self.trend_curve = self.plot_widget.plot(
            pen=pg.mkPen('#ff7f0e', width=2), 
            fillLevel=0, 
            brush=(255, 127, 14, 50) 
        )

    def start_parsing(self, data_dir, threshold, factor):
        """由主窗口调用，传入真实的调参结果，开始后台解析"""
        self.btn_load_log.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_info.setText("Parsing logs...")
        self.lbl_info.setStyleSheet("color: blue;")
        
        # 启动后台解析线程
        self.parser_thread = LogParserThread(data_dir, threshold, factor)
        self.parser_thread.progress_update.connect(self.update_progress)
        self.parser_thread.data_ready.connect(self.on_data_ready)
        self.parser_thread.error_msg.connect(self.on_error)
        self.parser_thread.start()
    
    def update_progress(self, current, total, file_name):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_info.setText(f"Processing: {file_name} ({current}/{total})")

    def on_data_ready(self, cumulative_data, total_mass):
        self.trend_curve.setData(cumulative_data)
        self.lbl_info.setText(f"Analysis Complete! Total Sand Produced: {total_mass:.4f} g")
        self.lbl_info.setStyleSheet("color: green; font-weight: bold;")
        self.btn_load_log.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_error(self, msg):
        self.lbl_info.setText(f"Error: {msg}")
        self.lbl_info.setStyleSheet("color: red; font-weight: bold;")
        self.btn_load_log.setEnabled(True)
        self.progress_bar.setVisible(False)