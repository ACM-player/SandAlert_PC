# =============================================================================
# SandAlert PC - 主窗口
# =============================================================================
# 说明：本文件是 SandAlert 上位机的主窗口，负责：
#   1. 使用 QStackedWidget 实现"实时监测"与"历史日志"双模式切换
#   2. 通过菜单栏的 Operation 菜单切换两种模式
#   3. Setup -> Comm Settings... 弹出 SetupDialog，支持两种数据源：
#      a) 串口模式：连接真实硬件或虚拟串口（mock_hardware.py）
#      b) 文件回放模式：从本地 .txt 文件回放数据（无硬件时的替代方案）
#   4. 统一管理阈值绑定、标定流程、报警阈值等信号/槽连接
#   5. closeEvent 安全终止所有后台线程，防止 Python 崩溃
# =============================================================================

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog, QStackedWidget
from PySide6.QtGui import QActionGroup, QAction

from ui.tabs.signal_traces import SignalTracesTab
from ui.tabs.parameters import ParametersTab
from ui.datalog_view import DataLogWidget
from ui.dialogs.setup_dialog import SetupDialog  # 通信设置对话框
from core.comm import (
    EdgeGatewayThread, FileReplayThread, SerialReaderThread, ModbusReaderThread,
)


class MainWindow(QMainWindow):
    """SandAlert PC 主窗口：含实时监测和历史日志双模式"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SandAlert PC - Advanced Localization Version")
        self.resize(1000, 700)
        
        # ====== 当前活跃的通信线程引用（用于 closeEvent 安全关闭） ======
        self.active_thread = None
        
        # ====== 使用 QStackedWidget 作为主容器，实现双模式无缝切换 ======
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # ---- 模式1：实时监测面板 (索引 0) ----
        self.main_tabs = QTabWidget()
        self.signal_tab = SignalTracesTab()
        self.param_tab = ParametersTab()
        self.main_tabs.addTab(self.signal_tab, "Signal Traces")
        self.main_tabs.addTab(self.param_tab, "Parameters")
        
        # ---- 模式2：历史日志面板 (索引 1) ----
        self.datalog_view = DataLogWidget()
        
        self.stacked_widget.addWidget(self.main_tabs)    # Index 0
        self.stacked_widget.addWidget(self.datalog_view) # Index 1
        
        self.init_menu_bar()
        self.bind_signals()

    def bind_signals(self):
        """统一管理内部的信号与槽绑定"""
        # ---- 阈值绑定 ----
        self.param_tab.p160_input.textChanged.connect(self.on_threshold_changed)
        self.on_threshold_changed(self.param_tab.p160_input.text())
        
        # ---- 标定系统绑定 ----
        self.param_tab.btn_start_calib.clicked.connect(self.start_calibration)
        self.param_tab.btn_stop_calib.clicked.connect(self.stop_calibration)
        self.param_tab.factor_input.textChanged.connect(self.on_factor_changed)
        self.signal_tab.calibration_progress.connect(self.on_calibration_progress)

        # ---- 报警阈值绑定 ----
        self.param_tab.caution_level_input.textChanged.connect(self.on_alarm_levels_changed)
        self.param_tab.alarm_level_input.textChanged.connect(self.on_alarm_levels_changed)

        # ---- 历史日志分析按钮绑定（双核导入） ----
        # 按钮 A: CSV 文件（已在 DataLogWidget 内部绑定 _on_load_csv）
        # 按钮 B: TXT 文件夹（已在 DataLogWidget 内部绑定 _on_load_folder）
        # 两个按钮的 click 信号均已直接在 DataLogWidget 构造时连接

    def init_menu_bar(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        
        # File 菜单
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Exit", self.close)
        
        # Operation 菜单
        op_menu = menubar.addMenu("Operation")
        self.op_group = QActionGroup(self)
        
        self.act_realtime = QAction("SandAlert PC", self, checkable=True)
        self.act_realtime.setChecked(True)
        self.act_datalog = QAction("DataLogging", self, checkable=True)
        
        self.op_group.addAction(self.act_realtime)
        self.op_group.addAction(self.act_datalog)
        
        op_menu.addAction(self.act_realtime)
        op_menu.addAction(self.act_datalog)
        
        self.act_realtime.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.act_datalog.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        # Setup 菜单：弹出通信设置对话框
        setup_menu = menubar.addMenu("Setup")
        setup_action = QAction("Comm Settings...", self)
        setup_action.triggered.connect(self.show_setup_dialog)
        setup_menu.addAction(setup_action)
        
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About")

    # ==================== 回调函数 ====================

    def on_threshold_changed(self, text):
        """当用户在 Parameters 页面修改阈值时，实时同步到 signal_tab"""
        try:
            new_th = float(text)
            self.signal_tab.user_threshold = new_th
        except ValueError:
            pass

    def on_factor_changed(self, text):
        """当用户手动修改 Factor 时，实时同步到 signal_tab"""
        try:
            new_factor = float(text)
            if new_factor > 0:
                self.signal_tab.K_v2 = new_factor
        except ValueError:
            pass

    def on_alarm_levels_changed(self):
        """当用户修改报警阈值时，实时同步到 signal_tab"""
        try:
            new_caution = float(self.param_tab.caution_level_input.text())
            new_alarm = float(self.param_tab.alarm_level_input.text())
            self.signal_tab.caution_limit = new_caution
            self.signal_tab.alarm_limit = new_alarm
        except ValueError:
            pass

    def start_calibration(self):
        """开始标定：清空能量池，进入累加模式"""
        self.signal_tab.is_calibrating = True
        self.signal_tab.accumulated_energy = 0.0
        self.signal_tab.accumulated_impacts = 0.0
        self.signal_tab.calibration_sample_count = 0
        self.param_tab.btn_start_calib.setEnabled(False)
        self.param_tab.btn_stop_calib.setEnabled(True)
        self.param_tab.lbl_calib_status.setText(
            "Status: Accumulating... Total Impacts: 0 | Energy: 0.000000"
        )
        self.param_tab.lbl_calib_status.setStyleSheet("color: red; font-weight: bold;")

    def on_calibration_progress(
        self, total_impacts, total_energy, sample_count, source
    ):
        """显示来自当前活动数据源的实时标定累加进度。"""
        if not self.signal_tab.is_calibrating:
            return
        self.param_tab.lbl_calib_status.setText(
            f"Status: Accumulating [{source}] | Samples: {sample_count} | "
            f"Total Impacts: {total_impacts:.0f} | Energy: {total_energy:.6f}"
        )

    def stop_calibration(self):
        """停止标定：计算出专属转换系数并应用"""
        self.signal_tab.is_calibrating = False
        self.param_tab.btn_start_calib.setEnabled(True)
        self.param_tab.btn_stop_calib.setEnabled(False)
        try:
            injected_mass = float(self.param_tab.calib_mass_input.text())
            total_energy = self.signal_tab.accumulated_energy
            total_impacts = self.signal_tab.accumulated_impacts
            if injected_mass <= 0:
                raise ValueError
            if total_energy <= 0 and total_impacts <= 0:
                self.param_tab.lbl_calib_status.setText("Status: Failed. No sand impacts detected.")
                self.param_tab.lbl_calib_status.setStyleSheet("color: orange;")
                return
            # 正常波形/Server 分析优先使用声学能量；仅有 impact_rate 的
            # Modbus/兼容数据包则以累计撞击数作为可用的标定量。
            calibration_value = total_energy if total_energy > 0 else total_impacts
            metric_name = "Energy" if total_energy > 0 else "Impacts"
            new_factor = calibration_value / injected_mass
            self.param_tab.factor_input.setText(f"{new_factor:.6f}")
            self.param_tab.lbl_calib_status.setText(
                f"Status: Success! Factor Applied from {metric_name}. "
                f"Total Impacts: {total_impacts:.0f} | Total Energy: {total_energy:.6f}"
            )
            self.param_tab.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
        except ValueError:
            self.param_tab.lbl_calib_status.setText("Status: Error. Invalid Mass Input.")

    def show_setup_dialog(self):
        """
        弹出通信设置对话框，支持三种数据源：
        
        1. Mock String Mode（默认）：
           - 使用 SerialReaderThread 从虚拟串口读取声学波形数据
           - 用于无硬件时的算法调试
        
        2. Modbus RTU Mode：
           - 使用 ModbusReaderThread 通过 Modbus 协议读取仪表全量参数
           - 支持 RTU（真实串口）和 TCP（本地测试）两种子模式
        
        3. 文件回放模式：
           - 使用 FileReplayThread 从本地 .txt 文件回放数据
           - 无硬件时的替代方案
        """
        dialog = SetupDialog(self)
        if dialog.exec():
            # 用户点击了 "Save & Connect"
            
            # ---- 先停止当前正在运行的任何线程 ----
            self.stop_active_thread()
            
            # ---- 获取用户在对话框中设置的值 ----
            port = dialog.com_combo.currentText()
            baudrate = dialog.baud_combo.currentText()
            use_file_replay = dialog.file_replay_mode
            use_modbus = dialog.modbus_mode
            use_edge = dialog.edge_mode
            use_tcp = dialog.modbus_tcp_check.isChecked() if use_modbus else False
            
            if use_edge:
                edge_url = dialog.edge_url_input.text().strip()
                print(f"\n[调试] Edge Service 模式，连接 {edge_url}")
                self.active_thread = EdgeGatewayThread(edge_url)
                self.active_thread.waveform_received.connect(
                    self.signal_tab.update_edge_waveform
                )
                self.active_thread.event_received.connect(
                    self.signal_tab.update_from_edge_event
                )
                self.active_thread.status_msg.connect(
                    lambda msg: print(f"EDGE: {msg}")
                )
                self.signal_tab.set_edge_mode(True)
                self.active_thread.start()

            elif use_file_replay:
                self.signal_tab.set_edge_mode(False)
                # ====== 文件回放模式 ======
                data_dir = QFileDialog.getExistingDirectory(
                    self, "选择存放现场数据文件的文件夹", "/"
                )
                if data_dir:
                    print(f"\n[调试] 文件回放模式，选中路径: {data_dir}")
                    self.active_thread = FileReplayThread(data_dir)
                    self.active_thread.data_received.connect(self.signal_tab.update_from_serial)
                    self.active_thread.status_msg.connect(lambda msg: print(f"SYS MSG: {msg}"))
                    self.active_thread.start()
                    
            elif use_modbus:
                self.signal_tab.set_edge_mode(False)
                # ====== Modbus RTU/TCP 模式 ======
                if use_tcp:
                    # TCP 模式：连接本地 Mock 服务器
                    tcp_host = dialog.tcp_host_input.text().strip()
                    tcp_port = int(dialog.tcp_port_input.text().strip())
                    print(f"\n[调试] Modbus TCP 模式，连接 {tcp_host}:{tcp_port}")
                    self.active_thread = ModbusReaderThread(
                        port_or_host=tcp_host,
                        use_tcp=True,
                        tcp_port=tcp_port,
                    )
                else:
                    # RTU 模式：通过串口连接真实仪表
                    print(f"\n[调试] Modbus RTU 模式，串口 {port}，波特率 {baudrate}")
                    self.active_thread = ModbusReaderThread(
                        port_or_host=port,
                        use_tcp=False,
                    )
                
                # 连接信号：Modbus 发射 dict，UI 层通过类型检测处理
                self.active_thread.data_received.connect(self.signal_tab.update_from_serial)
                self.active_thread.status_msg.connect(lambda msg: print(f"MODBUS: {msg}"))
                self.active_thread.start()
                
            else:
                self.signal_tab.set_edge_mode(False)
                # ====== Mock String 串口模式（默认） ======
                print(f"\n[调试] Mock String 串口模式，连接 {port}，波特率 {baudrate}")
                self.active_thread = SerialReaderThread(port, baudrate)
                self.active_thread.data_received.connect(self.signal_tab.update_from_serial)
                self.active_thread.status_msg.connect(lambda msg: print(f"SERIAL: {msg}"))
                self.active_thread.start()

    
    def stop_active_thread(self):
        """
        安全停止当前活跃的通信线程。
        在切换数据源或关闭窗口时调用。
        """
        if self.active_thread is not None and self.active_thread.isRunning():
            print("[调试] 正在停止当前通信线程...")
            self.active_thread.stop()
            self.active_thread = None
    
    # ==================== 安全关闭：优雅终止所有后台线程 ====================
    def closeEvent(self, event):
        """
        重写窗口关闭事件，确保在 UI 销毁前安全终止所有后台线程。
        """
        print("[Shutdown] 正在安全关闭所有后台线程...")

        # 1. 停止活跃的通信线程（串口或文件回放）
        self.stop_active_thread()

        # 2. 停止历史日志解析线程（TXT 文件夹后台解析）
        self.datalog_view.stop_parser()

        # 3. 停止 TDOA 后台工作线程（SignalTracesTab）
        if hasattr(self, 'signal_tab') and hasattr(self.signal_tab, 'stop_all_workers'):
            self.signal_tab.stop_all_workers()

        # 4. 接受关闭事件
        print("[Shutdown] 所有后台线程已安全终止，窗口即将关闭。")
        event.accept()
