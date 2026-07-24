# from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
#                                QGroupBox, QFormLayout, QLineEdit, QPushButton)

# from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
#                                QGroupBox, QFormLayout, QLineEdit, QPushButton, QLabel)
# from PySide6.QtCore import Qt

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QGroupBox, QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox)
from PySide6.QtCore import Qt
from ui.tabs.development import DevelopmentTab

class ParametersTab(QWidget):
    """标签页 B: 参数设置"""
    def __init__(self, development_enabled=True, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.inner_tabs = QTabWidget()
        
        self.tab_relays = QWidget()
        self.tab_display = QWidget()
        self.tab_output = QWidget()
        self.tab_calibration = QWidget()
        self.tab_system = QWidget()
        self.development_tab = (
            DevelopmentTab() if development_enabled else None
        )
        
        self.setup_system_tab()
        self.setup_calibration_tab()
        self.setup_relays_tab() # ====== 新增：初始化标定页签 ======
        
        self.inner_tabs.addTab(self.tab_calibration, "Calibration") # 提到前面来方便调试
        self.inner_tabs.addTab(self.tab_system, "System")
        self.inner_tabs.addTab(self.tab_relays, "Relays")
        self.inner_tabs.addTab(self.tab_display, "Display")
        self.inner_tabs.addTab(self.tab_output, "Output")
        if self.development_tab is not None:
            self.inner_tabs.addTab(self.development_tab, "开发测试")
        
        self.layout.addWidget(self.inner_tabs)
        
    def setup_system_tab(self):
        """配置 System 子页签"""
        layout = QVBoxLayout(self.tab_system)
        group_box = QGroupBox("Core Monitoring Parameters")
        form_layout = QFormLayout()
        
        # ====== 黄金阈值：0.008V（与 signal_traces.py 保持一致） ======
        self.p160_input = QLineEdit("0.008") # 统一标定为 0.008V，过滤湍流噪声
        self.p160_input.setPlaceholderText("Enter threshold voltage...")
        form_layout.addRow("Threshold (P160) [V]:", self.p160_input)
        
        self.p_no_input = QLineEdit()
        self.p_val_input = QLineEdit()
        self.btn_read = QPushButton("Read")
        self.btn_send = QPushButton("Send")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_read)
        btn_layout.addWidget(self.btn_send)
        
        form_layout.addRow("Parameter No.:", self.p_no_input)
        form_layout.addRow("Value:", self.p_val_input)
        form_layout.addRow("", btn_layout)
        
        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addStretch()

    def setup_calibration_tab(self):
        """配置 Calibration 标定子页签"""
        layout = QVBoxLayout(self.tab_calibration)
        
        group_box = QGroupBox("Sand Injection Calibration (P600-P610)")
        form_layout = QFormLayout()
        
        # 1. 用户输入的标定砂量
        self.calib_mass_input = QLineEdit("100")
        form_layout.addRow("Injected Sand Mass [Grams]:", self.calib_mass_input)
        
        # 2. 标定控制按钮
        self.btn_start_calib = QPushButton("Start Calibration Accumulation")
        self.btn_start_calib.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_stop_calib = QPushButton("Stop & Calculate Factor")
        self.btn_stop_calib.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        self.btn_stop_calib.setEnabled(False) # 初始禁用
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start_calib)
        btn_layout.addWidget(self.btn_stop_calib)
        form_layout.addRow("Control:", btn_layout)
        
        # 3. 状态与结果显示
        self.lbl_calib_status = QLabel("Status: Ready")
        self.lbl_calib_status.setStyleSheet("color: blue;")
        form_layout.addRow("Status:", self.lbl_calib_status)
        
        self.factor_input = QLineEdit("1.2") # 对应底层的 K_v2
        self.factor_input.setStyleSheet("background-color: #ffffcc;") # 标黄显示这是重要参数
        form_layout.addRow("Current Factor (K*v^2):", self.factor_input)
        
        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addStretch()

    def setup_relays_tab(self):
        """配置 Relays 继电器与报警子页签"""
        layout = QVBoxLayout(self.tab_relays)
        
        group_box = QGroupBox("Alarm & Relay Configuration (P300-P320)")
        form_layout = QFormLayout()
        
        # 1. 报警阈值设定 (对应真实物理出砂量 g/s)
        self.caution_level_input = QLineEdit("0.05")
        self.alarm_level_input = QLineEdit("0.10")
        
        form_layout.addRow("Caution Level [g/s]:", self.caution_level_input)
        form_layout.addRow("Alarm Level (Critical) [g/s]:", self.alarm_level_input)
        
        # 2. 继电器物理映射
        self.relay1_combo = QComboBox()
        self.relay1_combo.addItems(["Disabled", "Caution", "Alarm", "System Fault"])
        self.relay1_combo.setCurrentText("Caution") # 继电器1默认绑给警告
        
        self.relay2_combo = QComboBox()
        self.relay2_combo.addItems(["Disabled", "Caution", "Alarm", "System Fault"])
        self.relay2_combo.setCurrentText("Alarm")   # 继电器2默认绑给严重报警
        
        form_layout.addRow("Relay 1 Assignment:", self.relay1_combo)
        form_layout.addRow("Relay 2 Assignment:", self.relay2_combo)
        
        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addStretch()
