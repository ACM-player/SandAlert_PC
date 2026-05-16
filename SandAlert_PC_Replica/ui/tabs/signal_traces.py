import numpy as np
from scipy.signal import find_peaks
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QCheckBox, QLineEdit, QProgressBar, QPushButton,
                               QFileDialog, QSplitter, QGroupBox)
from PySide6.QtCore import QTimer, Qt, QRectF
from PySide6.QtGui import QPen
import csv               # 用于 CSV 文件写入（工业报表数据持久化）
from datetime import datetime  # 用于生成时间戳和默认文件名
from core.dsp_engine import DSPEngine
from core.tdoa_worker import TDOAWorker

class SignalTracesTab(QWidget):
    """标签页 A: 实时波形图 + 三相分离器效率评估 + CSV 工业报表 + TDOA 阵列定位"""
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
        
        # ==================== 主分割器：左侧波形图 + 右侧极坐标图 ====================
        pg.setConfigOptions(antialias=True)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # ---- 左侧：波形图 ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Voltage', units='V')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
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
        
        # ---- 右侧：极坐标雷达图（TDOA方位角显示） ----
        self.polar_widget = pg.PlotWidget()
        self.polar_widget.setBackground('w')
        self.polar_widget.setLabel('left', '')
        self.polar_widget.setLabel('bottom', '')
        self.polar_widget.setTitle('TDOA-AE Azimuth (Polar)')
        # 锁定等比例，范围 [-1.2, 1.2] 以显示完整极坐标
        self.polar_widget.setAspectLocked(lock=True, ratio=1.0)
        self.polar_widget.setRange(xRange=[-1.3, 1.3], yRange=[-1.3, 1.3])
        self.polar_widget.hideButtons()
        self.polar_widget.showGrid(x=False, y=False)
        
        # 绘制极坐标网格（背景项）
        self._draw_polar_grid()
        
        # 极坐标散点：TDOA 方位角标记点（红色闪烁点）
        self.polar_hits_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen('r', width=2),
            brush=pg.mkBrush(255, 50, 50, 180),
            symbol='o',
            name='TDOA Hits'
        )
        self.polar_widget.addItem(self.polar_hits_scatter)
        
        # 最近一次 TDOA 结果的标记（更大、更亮的点 + 标签）
        self.polar_last_hit = pg.ScatterPlotItem(
            size=20,
            pen=pg.mkPen('#ff4500', width=3),
            brush=pg.mkBrush(255, 69, 0, 220),
            symbol='star',
            name='Latest TDOA'
        )
        self.polar_widget.addItem(self.polar_last_hit)
        
        # TDOA 方位角文字标签
        self.polar_angle_label = pg.TextItem(
            text="",
            anchor=(0.5, 1.2),
            color=(200, 0, 0),
            fill=pg.mkBrush(255, 255, 255, 180)
        )
        self.polar_widget.addItem(self.polar_angle_label)
        
        # 将左右组件加入分割器
        self.main_splitter.addWidget(self.plot_widget)
        self.main_splitter.addWidget(self.polar_widget)
        self.main_splitter.setSizes([600, 350])  # 左侧宽，右侧窄
        
        self.layout.addWidget(self.main_splitter)
        
        # ==================== 底部控制区 ====================
        self.control_layout = QHBoxLayout()
        
        # ---- 左侧：现有波形显示控制 ----
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
        
        # ---- TDOA 控制区域 ----
        self.control_layout.addWidget(QLabel("  |  "))
        self.cb_tdoa = QCheckBox("开启阵列空间定位 (TDOA-AE)")
        self.cb_tdoa.setChecked(False)
        self.cb_tdoa.setStyleSheet("color: #c0392b; font-weight: bold;")
        self.cb_tdoa.toggled.connect(self.on_tdoa_toggled)
        self.control_layout.addWidget(self.cb_tdoa)
        
        # TDOA 状态标签
        self.tdoa_status_label = QLabel("TDOA: 待机")
        self.tdoa_status_label.setStyleSheet("color: #888; font-style: italic;")
        self.control_layout.addWidget(self.tdoa_status_label)
        
        # ---- 右侧：十字光标坐标显示 ----
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.crosshair_label)
        self.layout.addLayout(self.control_layout)
        
        # ==================== TDOA 方位角历史缓存 ====================
        # 用于极坐标图显示最近 N 次 TDOA 估计结果（环形缓冲）
        self.tdoa_history = []          # 存储历史角度（度）
        self.TDOA_HISTORY_MAX = 50      # 最多保留 50 个点
        self.tdoa_last_angle = None     # 最近一次角度
        
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
        
        # ====== 多通道环形缓冲池（用于 TDOA 切片） ======
        # 由于当前数据源是单通道，我们维护一个 4 通道缓冲池
        # 当检测到撞击峰值时，模拟生成 4 通道阵列数据用于演示
        self.MULTI_CH_BUF_LEN = 3000
        self.multi_channel_buffer = np.full(
            (4, self.MULTI_CH_BUF_LEN), -2.85, dtype=np.float32
        )
        
        # ====== TDOA 异步工作线程 ======
        self.tdoa_worker = TDOAWorker(
            fs=2_000_000,   # TDOA 引擎采样率 2 MHz
            radius=0.05,    # 阵列半径 5 cm
            c=1500.0,       # 声速 1500 m/s
        )
        # 连接信号
        self.tdoa_worker.doa_result.connect(self.on_tdoa_result)
        self.tdoa_worker.doa_error.connect(self.on_tdoa_error)
        # 注意：tdoa_worker 在需要时由 on_tdoa_toggled 启动/停止
    
    # ================================================================
    #  极坐标网格绘制
    # ================================================================
    def _draw_polar_grid(self):
        """绘制极坐标雷达图的背景网格（同心圆 + 径向线）。"""
        # 同心圆
        for r in [0.25, 0.5, 0.75, 1.0]:
            circle = pg.QtWidgets.QGraphicsEllipseItem(
                QRectF(-r, -r, 2 * r, 2 * r)
            )
            circle.setPen(pg.mkPen(200, 200, 200, width=1, style=Qt.DashLine))
            circle.setZValue(-10)
            self.polar_widget.addItem(circle)
        
        # 径向线（每 30° 一条）
        for angle_deg in range(0, 360, 30):
            angle_rad = np.deg2rad(angle_deg)
            x = np.cos(angle_rad)
            y = np.sin(angle_rad)
            line = pg.PlotDataItem(
                [0, x], [0, y],
                pen=pg.mkPen(200, 200, 200, width=1, style=Qt.DashLine)
            )
            line.setZValue(-10)
            self.polar_widget.addItem(line)
            
            # 方位角标注（在 1.1 倍半径处）
            label_r = 1.15
            lx = label_r * np.cos(angle_rad)
            ly = label_r * np.sin(angle_rad)
            label = pg.TextItem(
                text=f"{angle_deg}°",
                color=(150, 150, 150),
                anchor=(0.5, 0.5)
            )
            label.setPos(lx, ly)
            self.polar_widget.addItem(label)
        
        # 标注四个基本方位（N, E, S, W）
        directions = [("0° (E)", 1.3, 0), ("90° (N)", 0, 1.3),
                      ("180° (W)", -1.3, 0), ("270° (S)", 0, -1.3)]
        for text, x, y in directions:
            lbl = pg.TextItem(text=text, color=(120, 120, 120), anchor=(0.5, 0.5))
            lbl.setPos(x, y)
            self.polar_widget.addItem(lbl)

    # ================================================================
    #  TDOA 开关回调
    # ================================================================
    def on_tdoa_toggled(self, enabled):
        """
        当用户勾选/取消「开启阵列空间定位」时调用。
        - 开启：启动 TDOA 工作线程
        - 关闭：停止 TDOA 工作线程，清空极坐标图
        """
        if enabled:
            if not self.tdoa_worker.isRunning():
                self.tdoa_worker.start()
                self.tdoa_status_label.setText("TDOA: 运行中")
                self.tdoa_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            print("[TDOA] 阵列空间定位已开启")
        else:
            if self.tdoa_worker.isRunning():
                self.tdoa_worker.stop()
                self.tdoa_status_label.setText("TDOA: 已关闭")
                self.tdoa_status_label.setStyleSheet("color: #888; font-style: italic;")
            # 清空极坐标图
            self.tdoa_history.clear()
            self.tdoa_last_angle = None
            self.polar_hits_scatter.setData([])
            self.polar_last_hit.setData([])
            self.polar_angle_label.setText("")
            print("[TDOA] 阵列空间定位已关闭")

    # ================================================================
    #  TDOA 结果回传槽函数（从异步线程安全调用）
    # ================================================================
    def on_tdoa_result(self, est_angle_deg, cost_min, tdoa_obs):
        """
        接收 TDOAWorker 计算完成的方位角结果，更新极坐标图。
        此函数在主 UI 线程执行（PySide6 Signal 机制保证）。
        
        参数
        ----------
        est_angle_deg : float
            估计的方位角（度）。
        cost_min : float
            代价函数最小值（越小越可靠）。
        tdoa_obs : object
            观测时延向量。
        """
        # ---- 更新历史记录 ----
        self.tdoa_last_angle = est_angle_deg
        self.tdoa_history.append(est_angle_deg)
        if len(self.tdoa_history) > self.TDOA_HISTORY_MAX:
            self.tdoa_history.pop(0)
        
        # ---- 更新极坐标散点 ----
        # 历史点：转换为极坐标 (r=1.0, θ=angle)
        if self.tdoa_history:
            hist_rad = np.deg2rad(self.tdoa_history)
            hist_x = np.cos(hist_rad)
            hist_y = np.sin(hist_rad)
            
            # 历史点用半透明小圆点
            spots_hist = [{
                'pos': (hx, hy),
                'size': 10,
                'pen': pg.mkPen(100, 150, 255, 120, width=1),
                'brush': pg.mkBrush(100, 150, 255, 80),
                'symbol': 'o'
            } for hx, hy in zip(hist_x, hist_y)]
            self.polar_hits_scatter.setData(spots_hist)
        
        # 最近一次用醒目的星形标记
        last_rad = np.deg2rad(est_angle_deg)
        last_x = np.cos(last_rad)
        last_y = np.sin(last_rad)
        self.polar_last_hit.setData([{
            'pos': (last_x, last_y),
            'size': 20,
            'pen': pg.mkPen('#ff4500', width=2),
            'brush': pg.mkBrush(255, 69, 0, 200),
            'symbol': 'star'
        }])
        
        # ---- 更新角度标签 ----
        reliability = "High" if cost_min < 1e-10 else \
                      "Medium" if cost_min < 1e-8 else "Low"
        self.polar_angle_label.setPos(last_x * 0.6, last_y * 0.6)
        self.polar_angle_label.setText(
            f"{est_angle_deg:.1f}°\n({reliability})"
        )
        
        # ---- 更新状态栏 ----
        self.tdoa_status_label.setText(
            f"TDOA: {est_angle_deg:.1f}° | cost={cost_min:.2e}"
        )
        self.tdoa_status_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        
        # 打印详细调试信息
        print(f"[TDOA] 方位角: {est_angle_deg:.1f}°  "
              f"| 代价: {cost_min:.2e}  "
              f"| 队列剩余: {self.tdoa_worker.pending_tasks}")

    def on_tdoa_error(self, error_msg):
        """TDOA 计算错误回调。"""
        self.tdoa_status_label.setText(f"TDOA Error: {error_msg}")
        self.tdoa_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        print(f"[TDOA] 错误: {error_msg}")

    # ================================================================
    #  切窗打地鼠：生成 4 通道 Mock 阵列数据
    # ================================================================
    def _generate_mock_multi_ch_slice(self, peak_idx_in_window: int):
        """
        当检测到有效撞击峰值时，以该峰值为中心生成模拟的 4 通道阵列数据切片。
        
        由于当前数据源为单通道（串口/文件回放），此方法模拟 4 通道 UCA 接收，
        使用 TDOAEngine.generate_mock_array_data() 生成带物理时延的阵列数据。
        
        参数
        ----------
        peak_idx_in_window : int
            峰值在 self.voltage_data 窗口中的索引位置。
        
        返回
        -------
        X_slice : np.ndarray or None
            shape (4, n_slice) 的多通道切片；若失败则返回 None。
        """
        try:
            # 从 TDOAWorker 持有的引擎获取 Mock 数据生成能力
            # 使用一个随机方位角来模拟真实场景的多样性
            mock_angle = np.random.uniform(0, 360)
            
            # 切片长度：±0.5ms @ 2MHz = 2000 samples
            n_slice = 2000
            
            # 使用引擎生成模拟 4 通道数据
            X_mock, _ = self.tdoa_worker._engine.generate_mock_array_data(
                n_samples=n_slice,
                true_angle_deg=mock_angle,
                snr_db=8.0,
            )
            
            return X_mock
            
        except Exception as e:
            print(f"[TDOA] Mock 数据生成失败: {e}")
            return None

    # ==================== 鼠标十字光标 ====================
    def mouseMoved(self, evt):
        """鼠标移动时更新十字光标的位置和坐标显示"""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.crosshair_label.setText(f"X: {mousePoint.x():.2f} s, Y: {mousePoint.y():.2f} V")

    # ==================== 核心数据更新入口 ====================
    def update_from_serial(self, new_data):
        """
        接收 QThread 传来的真实一维 numpy 数组，送入 DSP 处理并更新图表。
        同时在此方法中完成：
          1. 三相分离器效率计算
          2. CSV 工业报表数据持久化
          3. 【新增】TDOA 切窗打地鼠：检测到撞击峰值时，生成阵列切片送入 TDOAWorker
        """
        shift = len(new_data)
        if shift == 0:
            return
            
        # ---- 滚动并拼装原始数据 ----
        self.voltage_data = np.roll(self.voltage_data, -shift)
        self.voltage_data[-shift:] = new_data
        
        # ---- 同时也更新多通道缓冲池（所有通道同步更新相同数据） ----
        self.multi_channel_buffer = np.roll(self.multi_channel_buffer, -shift, axis=1)
        self.multi_channel_buffer[:, -shift:] = new_data  # 4 通道初始相同
        
        # ---- 送入 DSP 引擎处理 ----
        # 【兼容】DSPEngine.process_window 现在返回 6 个值（新增 valid_peaks）
        processed_signal, threshold_limit, ips, mass_rate, window_energy, valid_peaks = self.dsp.process_window(
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
        try:
            fluid_flow_kg_s = float(self.fluid_flow_input.text())
        except (ValueError, AttributeError):
            fluid_flow_kg_s = 10.0
        
        if fluid_flow_kg_s > 0:
            efficiency = (mass_rate / 1000.0) / fluid_flow_kg_s * 100.0
        else:
            efficiency = 0.0
        
        efficiency = np.clip(efficiency, 0.0, 100.0)
        self.efficiency_bar.setValue(int(round(efficiency)))
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
            dt = shift / 1000.0
            step_mass = mass_rate * dt
            self.total_mass_recorded += step_mass
            
            mock_current_ma = 4.0 + (mass_rate * 0.1)
            
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            self.csv_writer.writerow([
                timestamp_str,
                ips,
                f"{mass_rate:.5f}",
                f"{efficiency:.2f}",
                f"{mock_current_ma:.2f}",
                f"{self.total_mass_recorded:.4f}"
            ])
        
        # ====== 【核心新增】TDOA 切窗打地鼠触发 ======
        if self.cb_tdoa.isChecked() and ips > 0 and valid_peaks:
            # 取最近一次有效峰值（窗口末尾的最新撞击）
            last_peak = valid_peaks[-1]
            
            # 使用峰值生成模拟 4 通道阵列数据切片
            X_slice = self._generate_mock_multi_ch_slice(last_peak)
            
            if X_slice is not None:
                # 异步投入 TDOA 工作线程
                self.tdoa_worker.enqueue(X_slice)
                
                # 如果队列积压过多，丢弃旧任务以保持实时性
                if self.tdoa_worker.pending_tasks > 5:
                    self.tdoa_worker.clear_queue()
                    self.tdoa_status_label.setText("TDOA: 清空积压队列")
        
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
            now = datetime.now()
            default_filename = f"SandLog_{now.strftime('%Y_%m_%d_%H_%M_%S')}.csv"
            
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "选择 CSV 文件保存路径",
                default_filename,
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
            
            try:
                self.csv_file = open(file_path, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                
                self.csv_writer.writerow([
                    "Timestamp",
                    "IPS",
                    "MassRate(g/s)",
                    "Efficiency(%)",
                    "Current(mA)",
                    "CumulativeMass(g)"
                ])
                
                self.is_recording = True
                self.total_mass_recorded = 0.0
                
                self.btn_csv_log.setText("Stop CSV Log")
                self.btn_csv_log.setStyleSheet(
                    "background-color: #dc3545; color: white; font-weight: bold; border-radius: 4px;"
                )
                
                print(f"[CSV Log] 开始录制，文件保存至: {file_path}")
                
            except Exception as e:
                print(f"[CSV Log] 文件创建失败: {str(e)}")
                if self.csv_file is not None:
                    self.csv_file.close()
                    self.csv_file = None
                    self.csv_writer = None
        
        else:
            # ---- 停止录制 ----
            self.is_recording = False
            
            if self.csv_file is not None:
                try:
                    self.csv_file.close()
                except Exception as e:
                    print(f"[CSV Log] 关闭文件时出错: {str(e)}")
                finally:
                    self.csv_file = None
                    self.csv_writer = None
            
            self.btn_csv_log.setText("Start CSV Log")
            self.btn_csv_log.setStyleSheet(
                "background-color: #17a2b8; color: white; font-weight: bold; border-radius: 4px;"
            )

    # ==================== 安全清理方法 ====================
    def stop_all_workers(self):
        """
        安全停止所有后台工作线程。
        由 MainWindow.closeEvent 在窗口关闭时调用。
        """
        # 停止 TDOA 工作线程
        if hasattr(self, 'tdoa_worker') and self.tdoa_worker.isRunning():
            print("[Shutdown] 正在停止 TDOA 工作线程...")
            self.tdoa_worker.stop()
            if not self.tdoa_worker.wait(2000):
                print("[Shutdown] 警告：TDOA 工作线程未能在 2 秒内退出")
