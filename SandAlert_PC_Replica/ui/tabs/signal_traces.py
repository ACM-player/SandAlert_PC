import time
import numpy as np
from scipy.signal import find_peaks

import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QCheckBox, QLineEdit, QProgressBar, QPushButton,
                               QFileDialog, QSplitter, QGroupBox)
from PySide6.QtCore import QTimer, Qt, QRectF
from PySide6.QtGui import QPen
import csv               # 用于 CSV 文件写入（工业报表数据持久化）
import traceback         # 用于 UI 槽函数内的异常打印诊断

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
        # 彻底隐藏无关的直角坐标轴，解决 UI 拥挤问题
        self.polar_widget.hideAxis('left')
        self.polar_widget.hideAxis('bottom')
        self.polar_widget.setTitle('TDOA-AE Azimuth (Polar)')
        # 锁定等比例，范围 [-1.3, 1.3] 给文字留出呼吸空间
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
        self._edge_mode = False         # True 时算法在远端服务器执行
        
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
        
        # ====== TDOA 限流抽帧（Rate Limiter） ======
        self.last_tdoa_time = 0.0           # 上次 TDOA 派发时间戳
        self.TDOA_MIN_INTERVAL = 0.05        # 触发抑制（秒）：一个物理脉冲只做一次最高质量切窗
        self.TDOA_QUEUE_WARN_THRESHOLD = 5  # 队列积压超过此值则清空

        # ====== TDOA 异步工作线程 ======
        self.tdoa_worker = TDOAWorker(
            fs=2_000_000,   # TDOA 引擎采样率 2 MHz
            radius=0.05,    # 阵列半径 5 cm
            c=1500.0,       # 声速 1500 m/s
        )
        # 连接信号
        self.tdoa_worker.doa_result.connect(self.on_tdoa_result)
        self.tdoa_worker.doa_error.connect(self.on_tdoa_error)
        self.tdoa_worker.doa_likelihood_low.connect(self.on_doa_likelihood_low)
        # 注意：tdoa_worker 在需要时由 on_tdoa_toggled 启动/停止

    
    # ================================================================
    #  极坐标网格绘制
    # ================================================================
    def _draw_polar_grid(self):
        """绘制清爽版的极坐标背景网格与标签"""
        # 画同心圆 (r=0.5, r=1.0)
        for r in [0.5, 1.0]:
            circle = pg.QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pg.mkPen('gray', width=1, style=Qt.DashLine))
            self.polar_widget.addItem(circle)
            
        # 统一的标签半径（推到外侧 1.25，留出呼吸空间）
        label_r = 1.25 
        
        # 统一循环画放射线与标签
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            x = np.cos(rad)
            y = np.sin(rad)
            
            # 画放射线 (只延伸到 r=1.0)
            line = pg.PlotCurveItem([0, x], [0, y], pen=pg.mkPen('gray', width=1, style=Qt.DashLine))
            self.polar_widget.addItem(line)
            
            # 整合标签文本（摒弃重复的单独绘制）
            if deg == 0:
                text = "0° (E)"
            elif deg == 90:
                text = "90° (N)"
            elif deg == 180:
                text = "180° (W)"
            elif deg == 270:
                text = "270° (S)"
            else:
                text = f"{deg}°"
                
            # 添加文本标签（统一锚点中心对齐）
            text_item = pg.TextItem(text, color='gray', anchor=(0.5, 0.5))
            text_item.setPos(x * label_r, y * label_r)
            self.polar_widget.addItem(text_item)

    # ================================================================
    #  TDOA 开关回调
    # ================================================================
    def on_tdoa_toggled(self, enabled):
        """
        当用户勾选/取消「开启阵列空间定位」时调用。
        - 开启：启动 TDOA 工作线程
        - 关闭：停止 TDOA 工作线程，清空极坐标图
        """
        if self._edge_mode:
            self.tdoa_status_label.setText(
                "TDOA: 等待服务器结果" if enabled else "TDOA: 远端显示已关闭"
            )
            return
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

    def on_doa_likelihood_low(self, cost_min: float):
        """
        当 TDOA 计算结果代价值过高（似然极低）时回调。
        不更新红星，在状态栏显示橙色警告。

        参数
        ----------
        cost_min : float
            最小代价值（超出似然门槛）。
        """
        self.tdoa_status_label.setText(
            f"⚠️ 杂波干扰 (Low Likelihood: {cost_min:.2e})"
        )
        self.tdoa_status_label.setStyleSheet(
            "color: #e67e22; font-weight: bold; font-style: italic;"
        )

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

    # ==================== Edge/Server 前后端分离入口 ====================
    def set_edge_mode(self, enabled: bool):
        """切换到独立 Edge 采集 + 服务器计算模式。"""
        self._edge_mode = enabled
        if enabled:
            if self.tdoa_worker.isRunning():
                self.tdoa_worker.stop()
            self.cb_tdoa.setChecked(True)
            self.tdoa_status_label.setText("TDOA: 等待服务器结果")
            self.btn_csv_log.setEnabled(False)
            self.btn_csv_log.setToolTip("Edge 模式下正式记录由服务器集中保存")
        else:
            self.btn_csv_log.setEnabled(True)
            self.btn_csv_log.setToolTip("")

    def update_edge_waveform(self, new_data):
        """仅在 Mac 本地绘制 Edge 发布的原始波形，不执行本地算法。"""
        if new_data.ndim == 2:
            shift = min(new_data.shape[1], self.data_len)
            ch1 = new_data[0, -shift:]
            self.voltage_data = np.roll(self.voltage_data, -shift)
            self.voltage_data[-shift:] = ch1
            if new_data.shape[0] == 4:
                self.multi_channel_buffer = np.roll(
                    self.multi_channel_buffer, -shift, axis=1
                )
                self.multi_channel_buffer[:, -shift:] = new_data[:, -shift:]
        else:
            shift = min(len(new_data), self.data_len)
            self.voltage_data = np.roll(self.voltage_data, -shift)
            self.voltage_data[-shift:] = new_data[-shift:]
        self.raw_curve.setData(self.voltage_data)

    def update_from_edge_event(self, event):
        """消费服务器通过 Edge 转发的分析、Modbus 和连接状态事件。"""
        event_type = event.get("type", "")
        if event_type == "analysis.result":
            data = event
            ips = int(data.get("ips", 0))
            mass_rate = float(data.get("mass_rate", 0.0))
            threshold = float(data.get("threshold", self.user_threshold))
            self.impact_label.setText(f"Impact rate = {ips} /s  [Server]")
            self.mass_label.setText(f"Mass rate = {mass_rate:.5f} g/s  [Server]")
            self.th_curve.setData(np.full(self.data_len, threshold))
            self._update_edge_efficiency(mass_rate)
            self._set_alarm_style(data.get("alarm_level", "normal"))
            angle = data.get("tdoa_angle_deg")
            if angle is not None and data.get("tdoa_reliable", False):
                self.on_tdoa_result(
                    float(angle), float(data.get("tdoa_cost") or 0.0), None
                )
        elif event_type == "tdoa.result":
            data = event.get("data", {})
            angle = data.get("angle_deg")
            if angle is not None and data.get("reliable", False):
                self.on_tdoa_result(
                    float(angle), float(data.get("cost") or 0.0), None
                )
        elif event_type == "modbus.result":
            data = event.get("data", {})
            self.impact_label.setText(
                f"Impact rate = {int(data.get('ips', 0))} /s  [Server Modbus]"
            )
            mass_rate = float(data.get("mass_rate", 0.0))
            self.mass_label.setText(
                f"Mass rate = {mass_rate:.5f} g/s  [Server Modbus]"
            )
            relay = int(data.get("relay_status", 0))
            level = "critical" if relay & 0x02 else "caution" if relay & 0x01 else "normal"
            self._set_alarm_style(level)
            self._update_edge_efficiency(mass_rate)
        elif event_type == "edge.connection":
            status = event.get("data", {}).get("status", "unknown")
            self.tdoa_status_label.setText(f"Server: {status}")
        elif event_type == "service.error":
            message = event.get("data", {}).get("message", "unknown error")
            self.tdoa_status_label.setText(f"Server Error: {message}")

    def _update_edge_efficiency(self, mass_rate: float):
        try:
            flow = float(self.fluid_flow_input.text())
        except ValueError:
            flow = 10.0
        efficiency = np.clip((mass_rate / 1000.0) / flow * 100.0 if flow > 0 else 0.0, 0, 100)
        self.efficiency_bar.setValue(int(round(efficiency)))
        self.efficiency_bar.setFormat(f"Efficiency: {efficiency:.4f}%")

    def _set_alarm_style(self, level: str):
        if level == "critical":
            text, background, color = "CRITICAL ALARM", "#dc3545", "white"
        elif level == "caution":
            text, background, color = "CAUTION", "#ffc107", "black"
        else:
            text, background, color = "NORMAL", "#28a745", "white"
        self.alarm_indicator.setText(text)
        self.alarm_indicator.setStyleSheet(
            f"background-color: {background}; color: {color}; "
            "font-weight: bold; border-radius: 5px;"
        )

    # ==================== 核心数据更新入口 ====================
    def update_from_serial(self, new_data):
        """
        接收 QThread 传来的数据，支持两种类型（自动检测）：
        
        类型 A — numpy 数组（来自 SerialReaderThread / FileReplayThread）:
            波形声学数据，走 DSP 引擎处理
            支持 shape (4, N) 4通道阵列 或 (N,) 单通道
        
        类型 B — dict（来自 ModbusReaderThread）:
            已解析的 Modbus 工业参数，直接使用
            dict 字段: ips, mass_rate, total_mass, relay_status, current_ma
        
        在本函数中统一完成：
          1. 三相分离器效率计算
          2. 报警状态机判定（正常/警告/严重）
          3. CSV 工业报表数据持久化
          4. 实时 UI 更新
        """
        # ================================================================
        #  类型检测：判断传入的是 dict（Modbus）还是 numpy array（波形）
        # ================================================================
        if isinstance(new_data, dict):
            # ====== Modbus 工业数据模式（直接使用解析后的参数） ======
            # 用 try/except 包裹整个处理逻辑，防止 PySide 静默吞噬异常
            try:
                print(f"[UI] 🧵 收到 Modbus 字典: {new_data}")
                ips = new_data.get('ips', 0)
                mass_rate = new_data.get('mass_rate', 0.0)
                total_mass = new_data.get('total_mass', 0.0)
                relay_status = new_data.get('relay_status', 0)
                current_ma = new_data.get('current_ma', 4.0)
                
                # Modbus 模式下，波形图维持不动（不更新 DSP），只更新数字参数
                # 取当前窗口均值作为"平稳基线"，但波形不刷新
                processed_signal = self.voltage_data  # 保持波形
                threshold_limit = self.user_threshold
                
                # Modbus 没有"窗口能量"概念，跳过标定累加
                shift = 50  # 给 CSV 一个合理的 dt 基值（0.05s）
                
                # ---- 更新状态显示 ----
                self.impact_label.setText(f"Impact rate = {ips} /s  [Modbus]")
                self.mass_label.setText(f"Mass rate = {mass_rate:.5f} g/s  [Modbus]")
            except Exception as e:
                print(f"[UI] ❌ Modbus 更新界面时发生崩溃: {e}")
                traceback.print_exc()
                return  # 跳过后续共用逻辑，避免级联错误
            
        else:
            # ====== 波形声学数据模式（原 DSP 流程） ======
            
            # ---- 检测数据格式 ----
            if new_data.ndim == 2 and new_data.shape[0] == 4:
                # ----- 4 通道阵列数据模式 -----
                n_frames = new_data.shape[1]
                if n_frames == 0:
                    return
                
                # 提取 Ch.1 供 DSP 引擎处理
                ch1_data = new_data[0]
                
                # 滚动并更新波形通道
                self.voltage_data = np.roll(self.voltage_data, -n_frames)
                self.voltage_data[-n_frames:] = ch1_data
                
                # 滚动并更新多通道缓冲池
                self.multi_channel_buffer = np.roll(
                    self.multi_channel_buffer, -n_frames, axis=1
                )
                self.multi_channel_buffer[:, -n_frames:] = new_data
                
                shift = n_frames
            else:
                # ----- 单通道模式（向后兼容） -----
                shift = len(new_data)
                if shift == 0:
                    return
                
                # 滚动并更新波形通道
                self.voltage_data = np.roll(self.voltage_data, -shift)
                self.voltage_data[-shift:] = new_data
                
                # 多通道缓冲池：所有通道填入相同数据
                self.multi_channel_buffer = np.roll(
                    self.multi_channel_buffer, -shift, axis=1
                )
                self.multi_channel_buffer[:, -shift:] = new_data
            
            # ---- 送入 DSP 引擎处理（使用 Ch.1 数据） ----
            processed_signal, threshold_limit, ips, mass_rate, window_energy, valid_peaks = (
                self.dsp.process_window(
                    self.voltage_data,
                    self.user_threshold,
                    self.K_v2
                )
            )
            
            # ---- 标定累加逻辑 ----
            if getattr(self, 'is_calibrating', False):
                self.accumulated_energy += window_energy
            
            # ---- 更新 UI 数字 ----
            self.impact_label.setText(f"Impact rate = {ips} /s")
            self.mass_label.setText(f"Mass rate = {mass_rate:.5f} g/s")
            
            # Modbus 模式下无电流数据时使用模拟值
            current_ma = 4.0 + (mass_rate * 0.1)
            relay_status = None
            
            # ====== 【Modbus 模式跳过】TDOA 切窗打地鼠（波形模式下执行） ======
            if self.cb_tdoa.isChecked() and ips > 0:
                # 全局能量重心准星
                global_max_idx = int(np.argmax(np.abs(self.voltage_data)))
                
                if 1450 <= global_max_idx <= 1550:
                    now = time.time()
                    if now - self.last_tdoa_time >= 0.1:
                        X_slice = self._extract_multi_ch_slice(global_max_idx)
                        if X_slice is not None:
                            self.last_tdoa_time = now
                            self.tdoa_worker.enqueue(X_slice)
                            
                            if self.tdoa_worker.pending_tasks > self.TDOA_QUEUE_WARN_THRESHOLD:
                                self.tdoa_worker.clear_queue()
                                self.tdoa_status_label.setText("TDOA: 清空积压队列")
            
            # ---- 波形模式绘图更新 ----
            self.raw_curve.setData(self.voltage_data)
            self.avg_curve.setData(processed_signal)
            self.th_curve.setData(np.full(self.data_len, threshold_limit))
        
        # ================================================================
        #  以下为两种模式共用的逻辑（效率计算、报警判定、CSV 持久化）
        # ================================================================
        
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
        
        # ====== 报警状态机判定（使用 Modbus relay_status 或 mass_rate 阈值） ======
        if isinstance(new_data, dict) and relay_status is not None:
            # Modbus 模式：基于仪表原生的继电器状态判定
            if relay_status & 0x02:  # bit1 = Alarm
                self.alarm_indicator.setText("CRITICAL ALARM")
                self.alarm_indicator.setStyleSheet(
                    "background-color: #dc3545; color: white; font-weight: bold; border-radius: 5px;"
                )
            elif relay_status & 0x01:  # bit0 = Caution
                self.alarm_indicator.setText("CAUTION")
                self.alarm_indicator.setStyleSheet(
                    "background-color: #ffc107; color: black; font-weight: bold; border-radius: 5px;"
                )
            else:
                self.alarm_indicator.setText("NORMAL")
                self.alarm_indicator.setStyleSheet(
                    "background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;"
                )
        else:
            # 波形模式：基于 DSP 计算的 mass_rate 阈值判定
            if mass_rate >= self.alarm_limit:
                self.alarm_indicator.setText("CRITICAL ALARM")
                self.alarm_indicator.setStyleSheet(
                    "background-color: #dc3545; color: white; font-weight: bold; border-radius: 5px;"
                )
            elif mass_rate >= self.caution_limit:
                self.alarm_indicator.setText("CAUTION")
                self.alarm_indicator.setStyleSheet(
                    "background-color: #ffc107; color: black; font-weight: bold; border-radius: 5px;"
                )
            else:
                self.alarm_indicator.setText("NORMAL")
                self.alarm_indicator.setStyleSheet(
                    "background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;"
                )
        
        # ====== CSV 工业报表数据持久化 ======
        if self.is_recording and self.csv_writer is not None:
            dt = shift / 1000.0
            step_mass = mass_rate * dt
            self.total_mass_recorded += step_mass
            
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 提取最新的 TDOA 角度（若未开启或无数据则填 N/A）
            if self.cb_tdoa.isChecked() and self.tdoa_last_angle is not None:
                tdoa_val = f"{self.tdoa_last_angle:.1f}"
            else:
                tdoa_val = "N/A"
            
            self.csv_writer.writerow([
                timestamp_str,
                ips,
                f"{mass_rate:.5f}",
                f"{efficiency:.2f}",
                f"{current_ma:.2f}",
                f"{self.total_mass_recorded:.4f}",
                tdoa_val
            ])


    # ================================================================
    #  从多通道缓冲池截取 4 通道切片（真实的"切窗打地鼠"）
    # ================================================================
    def _extract_multi_ch_slice(self, peak_idx_in_window: int) -> np.ndarray:
        """
        从 multi_channel_buffer 中以峰值位置为中心，截取极短的 4 通道切片。
        这是"切窗打地鼠"策略的核心——从连续的环形缓冲池中切出干净的事件片段。
        
        采用「向左平移对齐」策略：峰后最多取 1000 点（不超右边界），
        不足部分从前方补齐，保证切片长度绝对等于 2000。
        
        参数
        ----------
        peak_idx_in_window : int
            峰值在 self.voltage_data 窗口中的索引位置。
        
        返回
        -------
        X_slice : np.ndarray, shape (4, n_slice)
            多通道数据切片。若切片无效则返回 None。
        """
        n_slice = 2000     # 总长度 2000 点
        buf_len = self.MULTI_CH_BUF_LEN  # 3000
        
        # 尝试向未来取 1000 点，但不允许超过右边界
        end = min(peak_idx_in_window + 1000, buf_len)
        # 强制向左回退 n_slice，保证切片长度绝对等于 2000
        start = end - n_slice
        
        # 只有在软件刚启动，总数据量都不足 2000 时才放弃
        if start < 0:
            return None
        
        # 截取 4 通道切片
        X_slice = self.multi_channel_buffer[:, start:end].copy()
        
        return X_slice

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
                    "CumulativeMass(g)",
                    "TDOA_Angle(deg)"  # 新增定位角度列
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
