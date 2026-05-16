import numpy as np
from scipy.signal import find_peaks
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QCheckBox, QLineEdit, QProgressBar, QPushButton,
                               QFileDialog)
from PySide6.QtCore import QTimer, Qt
import csv               # 用于 CSV 文件写入（工业报表数据持久化）
from datetime import datetime  # 用于生成时间戳和默认文件名
from core.dsp_engine import DSPEngine

class SignalTracesTab(QWidget):
    """标签页 A: 实时波形图 + 三相分离器效率评估 + CSV 工业报表"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # ==================== 顶部信息栏 ====================
        self.info_layout = QHBoxLayout()
        
        # ---- 1. 工业警报指示灯 ----
        self.alarm_indicator = QLabel("NORMAL")
        self.alarm_indicator.setAlignment(Qt.AlignCenter)
        self.alarm_indicator.setFixedSize(120, 30)
        self.alarm_indicator.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;")
        self.info_layout.addWidget(self.alarm_indicator)
        
        # ---- 2. IPS 和 Mass Rate 标签 ----
        self.impact_label = QLabel("Impact rate = 0 /s")
        self.mass_label = QLabel("Mass rate = 0.000 g/s")
        self.info_layout.addWidget(self.impact_label)
        self.info_layout.addWidget(self.mass_label)
        
        # ---- 3. 三相分离器效率评估 UI ----
        # 流体流量输入框，默认 10.0 kg/s
        self.fluid_flow_input = QLineEdit("10.0")
        self.fluid_flow_input.setFixedWidth(80)
        self.info_layout.addWidget(QLabel("Fluid Flow [kg/s]:"))
        self.info_layout.addWidget(self.fluid_flow_input)
        
        # 分离效率进度条，范围 0-100，蓝色风格
        self.efficiency_bar = QProgressBar()
        self.efficiency_bar.setRange(0, 100)
        self.efficiency_bar.setValue(0)
        self.efficiency_bar.setFixedWidth(200)
        self.efficiency_bar.setFormat("Efficiency: 0%")
        self.efficiency_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; }"
            "QProgressBar::chunk { background-color: #007bff; }"
        )
        self.info_layout.addWidget(QLabel("Efficiency:"))
        self.info_layout.addWidget(self.efficiency_bar)
        
        # ---- 4. CSV 录制按钮（最右侧） ----
        self.btn_csv_log = QPushButton("Start CSV Log")
        self.btn_csv_log.setFixedWidth(120)
        self.btn_csv_log.setStyleSheet(
            "background-color: #17a2b8; color: white; font-weight: bold; border-radius: 4px;"
        )
        self.btn_csv_log.clicked.connect(self.toggle_csv_recording)
        self.info_layout.addStretch()
        self.info_layout.addWidget(self.btn_csv_log)
        
        self.layout.addLayout(self.info_layout)
        
        # ====== 报警阈值变量 ======
        self.caution_limit = 0.05
        self.alarm_limit = 0.10
        
        # ==================== PyQtGraph 绘图区 ====================
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Voltage', units='V')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)
        
        # 三条曲线：原始信号(红)、处理信号(蓝)、阈值线(绿)
        self.raw_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=1.5), name="Raw Signal")
        self.avg_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2, style=Qt.DashLine), name="Average")
        self.th_curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2, style=Qt.DashLine), name="Threshold")
        
        # 十字光标
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', style=Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', style=Qt.DashLine))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        
        self.crosshair_label = QLabel("X: 0.00 s, Y: 0.00 V")
        self.crosshair_label.setAlignment(Qt.AlignRight)
        
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        
        # ==================== 底部控制区 ====================
        self.control_layout = QHBoxLayout()
        self.cb_raw = QCheckBox("Show Raw Signal (Red)")
        self.cb_raw.setChecked(True)
        self.cb_avg = QCheckBox("Show Average (Blue)")
        self.cb_avg.setChecked(True)
        self.cb_th = QCheckBox("Show Threshold (Green)")
        self.cb_th.setChecked(True)
        
        self.cb_raw.toggled.connect(self.raw_curve.setVisible)
        self.cb_avg.toggled.connect(self.avg_curve.setVisible)
        self.cb_th.toggled.connect(self.th_curve.setVisible)
        
        self.control_layout.addWidget(self.cb_raw)
        self.control_layout.addWidget(self.cb_avg)
        self.control_layout.addWidget(self.cb_th)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.crosshair_label)
        self.layout.addLayout(self.control_layout)

        # ==================== 数据缓存与 DSP 引擎 ====================
        self.data_len = 3000  # 视窗大小：3000 个采样点
        self.time_data = np.arange(-self.data_len, 0)  # X 轴为相对采样点
        self.voltage_data = np.full(self.data_len, -2.85)  # 预填充接近真实的底噪值
        
        # ====== 黄金阈值：0.008V ======
        self.user_threshold = 0.008
        self.K_v2 = 1.2
        self.dsp = DSPEngine(sample_rate=1000)
        
        # ====== 标定状态变量 ======
        self.is_calibrating = False
        self.accumulated_energy = 0.0
        
        # ====== CSV 录制状态变量 ======
        self.is_recording = False       # 是否正在录制
        self.csv_file = None            # CSV 文件对象
        self.csv_writer = None          # CSV 写入器
        self.total_mass_recorded = 0.0  # 本次录制的累积出砂量（克）

    # ==================== 鼠标十字光标 ====================
    def mouseMoved(self, evt):
        """鼠标移动时更新十字光标的位置和坐标显示"""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.crosshair_label.setText(f"X: {mousePoint.x():.2f} s, Y: {mousePoint.y():.2f} V")

    # ==================== 模拟数据更新（已弃用，保留供参考） ====================
    def update_simulation(self):
        """模拟数据生成（仅用于无硬件时的调试，当前未启用）"""
        shift = 20
        self.time_data = np.roll(self.time_data, -shift)
        self.time_data[-shift:] = self.time_data[-shift-1] + np.linspace(0.01, 0.2, shift)
        
        self.voltage_data = np.roll(self.voltage_data, -shift)
        base_noise = np.random.normal(1.0, 0.1, shift)
        
        if np.random.rand() > 0.8:
            impact_idx = np.random.randint(0, shift)
            base_noise[impact_idx] += np.random.uniform(0.8, 2.0)
            
        self.voltage_data[-shift:] = base_noise
        
        sb = np.mean(self.voltage_data)
        threshold_limit = sb + self.user_threshold
        
        peaks, _ = find_peaks(self.voltage_data, height=threshold_limit, distance=5)
        
        ips = len(peaks)
        if ips > 0:
            sp = self.voltage_data[peaks]
            mass_rate = (1 / self.K_v2) * np.sum((sp - sb)**2)
        else:
            mass_rate = 0.0

        self.impact_label.setText(f"Impact rate = {ips} /s")
        self.mass_label.setText(f"Mass rate = {mass_rate:.3f} g/s")
        
        self.raw_curve.setData(self.time_data, self.voltage_data)
        self.avg_curve.setData(self.time_data, np.full(self.data_len, sb))
        self.th_curve.setData(self.time_data, np.full(self.data_len, threshold_limit))

    # ==================== 核心数据更新入口 ====================
    def update_from_serial(self, new_data):
        """
        接收 QThread 传来的真实一维 numpy 数组，送入 DSP 处理并更新图表。
        同时在此方法中完成：
          1. 三相分离器效率计算
          2. CSV 工业报表数据持久化
        """
        shift = len(new_data)
        if shift == 0:
            return
            
        # ---- 滚动并拼装原始数据 ----
        self.voltage_data = np.roll(self.voltage_data, -shift)
        self.voltage_data[-shift:] = new_data
        
        # ---- 送入 DSP 引擎处理 ----
        processed_signal, threshold_limit, ips, mass_rate, window_energy = self.dsp.process_window(
            self.voltage_data, 
            self.user_threshold, 
            self.K_v2
        )
        
        # ---- 标定累加逻辑 ----
        if getattr(self, 'is_calibrating', False):
            self.accumulated_energy += window_energy
        
        # ---- 更新 UI 数字 ----
        self.impact_label.setText(f"Impact rate = {ips} /s")
        self.mass_label.setText(f"Mass rate = {mass_rate:.5f} g/s")
        
        # ====== 三相分离器效率计算 ======
        # 从输入框读取流体流量（kg/s），需做异常保护防止空值报错
        try:
            fluid_flow_kg_s = float(self.fluid_flow_input.text())
        except (ValueError, AttributeError):
            fluid_flow_kg_s = 10.0  # 如果用户输入非法，默认使用 10.0 kg/s
        
        # 效率公式：Efficiency = (mass_rate(g/s) / 1000) / fluid_flow(kg/s) * 100
        # 先将 mass_rate 从 g/s 换算为 kg/s，再除以流体流量得到比值，最后乘以 100 转为百分比
        if fluid_flow_kg_s > 0:
            efficiency = (mass_rate / 1000.0) / fluid_flow_kg_s * 100.0
        else:
            efficiency = 0.0
        
        # 将效率限制在 0~100 之间，防止除零或异常数据导致进度条越界
        efficiency = np.clip(efficiency, 0.0, 100.0)
        self.efficiency_bar.setValue(int(round(efficiency)))
        # 保留 4 位小数显示，确保极小浮点数（如 0.0002%）也能精确呈现，体现精密仪器感
        self.efficiency_bar.setFormat(f"Efficiency: {efficiency:.4f}%")
        
        # ---- 报警状态机判定 ----
        if mass_rate >= self.alarm_limit:
            self.alarm_indicator.setText("CRITICAL ALARM")
            self.alarm_indicator.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; border-radius: 5px;")
        elif mass_rate >= self.caution_limit:
            self.alarm_indicator.setText("CAUTION")
            self.alarm_indicator.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold; border-radius: 5px;")
        else:
            self.alarm_indicator.setText("NORMAL")
            self.alarm_indicator.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;")
        
        # ====== CSV 工业报表数据持久化 ======
        if self.is_recording and self.csv_writer is not None:
            # 计算本次数据块的时间间隔（秒）：采样点数 / 采样率(1000Hz)
            dt = shift / 1000.0
            # 本次数据块产生的出砂质量（克）= 瞬时出砂率(g/s) × 时间间隔(s)
            step_mass = mass_rate * dt
            # 累加到本次录制的总出砂量
            self.total_mass_recorded += step_mass
            
            # 模拟生成 4-20mA 工业电流值：4mA 为底值，mass_rate 越大电流越大
            # 公式：4.0 + mass_rate * 0.1，确保在合理范围内
            mock_current_ma = 4.0 + (mass_rate * 0.1)
            
            # 获取当前时间戳（精确到毫秒）
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留毫秒
            
            # 写入一行 CSV 数据
            self.csv_writer.writerow([
                timestamp_str,
                ips,
                f"{mass_rate:.5f}",
                f"{efficiency:.2f}",
                f"{mock_current_ma:.2f}",
                f"{self.total_mass_recorded:.4f}"
            ])
        
        # ---- 绘图更新 ----
        self.raw_curve.setData(self.voltage_data)
        self.avg_curve.setData(processed_signal)
        self.th_curve.setData(np.full(self.data_len, threshold_limit))

    # ==================== CSV 录制控制 ====================
    def toggle_csv_recording(self):
        """
        切换 CSV 录制状态：
        - 如果当前未录制：弹出 Mac 原生文件保存对话框，选择保存路径后开始录制
        - 如果当前正在录制：关闭文件，停止录制
        """
        if not self.is_recording:
            # ---- 开始录制 ----
            # 生成默认文件名：SandLog_年_月_日_时_分_秒.csv
            now = datetime.now()
            default_filename = f"SandLog_{now.strftime('%Y_%m_%d_%H_%M_%S')}.csv"
            
            # 弹出 Mac 原生文件保存对话框
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "选择 CSV 文件保存路径",
                default_filename,
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return  # 用户取消了对话框
            
            try:
                # 打开文件并创建 CSV 写入器
                self.csv_file = open(file_path, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                
                # 写入 CSV 表头
                self.csv_writer.writerow([
                    "Timestamp",           # 时间戳（精确到毫秒）
                    "IPS",                 # 每秒撞击次数
                    "MassRate(g/s)",       # 瞬时出砂率（克/秒）
                    "Efficiency(%)",       # 三相分离器效率（百分比）
                    "Current(mA)",         # 模拟 4-20mA 工业电流值
                    "CumulativeMass(g)"    # 本次录制的累积出砂量（克）
                ])
                
                # 重置录制状态变量
                self.is_recording = True
                self.total_mass_recorded = 0.0
                
                # 更新按钮样式为红色"停止"状态
                self.btn_csv_log.setText("Stop CSV Log")
                self.btn_csv_log.setStyleSheet(
                    "background-color: #dc3545; color: white; font-weight: bold; border-radius: 4px;"
                )
                
                print(f"[CSV Log] 开始录制，文件保存至: {file_path}")
                
            except Exception as e:
                print(f"[CSV Log] 文件创建失败: {str(e)}")
                # 清理可能已打开的文件对象
                if self.csv_file is not None:
                    self.csv_file.close()
                    self.csv_file = None
                    self.csv_writer = None
        
        else:
            # ---- 停止录制 ----
            self.is_recording = False
            
            # 安全关闭 CSV 文件
            if self.csv_file is not None:
                try:
                    self.csv_file.close()
                except Exception as e:
                    print(f"[CSV Log] 关闭文件时出错: {str(e)}")
                finally:
                    self.csv_file = None
                    self.csv_writer = None
            
            # 更新按钮样式为青色"开始"状态
            self.btn_csv_log.setText("Start CSV Log")
            self.btn_csv_log.setStyleSheet(
                "background-color: #17a2b8; color: white; font-weight: bold; border-radius: 4px;"
            )
            
