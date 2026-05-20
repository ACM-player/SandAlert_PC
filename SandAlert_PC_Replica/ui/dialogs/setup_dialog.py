"""
SandAlert PC - 通信设置对话框
==============================
功能：让用户选择数据源类型和通信参数。

支持三种数据源模式：
  1. Mock String Mode（默认）：模拟字符串模式，连接虚拟串口
  2. Modbus RTU Mode：通过 Modbus RTU/TCP 协议连接 SandAlert 仪表
  3. File Replay Mode：从本地 .txt 文件回放数据（无硬件时的替代方案）

使用方法：
  点击菜单栏 Setup -> Comm Settings... 弹出此对话框。
  选择模式，填写相应参数后，点击 "Save & Connect" 即可。
"""

from PySide6.QtWidgets import (QDialog, QFormLayout, QComboBox, QPushButton,
                               QLabel, QVBoxLayout, QGroupBox, QRadioButton,
                               QCheckBox, QLineEdit)
from PySide6.QtCore import Qt

class SetupDialog(QDialog):
    """
    通信设置对话框：支持三种数据源模式。
    
    属性:
        com_combo: 串口号下拉框（串口模式和 Modbus RTU 模式共用）
        baud_combo: 波特率下拉框
        file_replay_mode: 布尔值，True=文件回放模式
        modbus_mode: 布尔值，True=Modbus RTU 模式
        use_tcp: 布尔值，True=Modbus TCP 模式（本地测试）
        tcp_host_input: TCP 主机地址输入框
        tcp_port_input: TCP 端口号输入框
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup - Communication Settings")
        self.setFixedSize(420, 350)
        
        # ====== 主布局 ======
        main_layout = QVBoxLayout(self)
        
        # ====== 数据源模式选择 ======
        mode_group = QGroupBox("Data Source Mode")
        mode_layout = QVBoxLayout()
        
        # Mock String 模式（默认选中）
        self.radio_serial = QRadioButton("Mock String Mode (Virtual Serial)")
        self.radio_serial.setChecked(True)
        self.radio_serial.toggled.connect(self.on_mode_changed)
        
        # Modbus RTU 模式
        self.radio_modbus = QRadioButton("Modbus RTU Mode (Real Hardware)")
        self.radio_modbus.toggled.connect(self.on_mode_changed)
        
        # 文件回放模式
        self.radio_file = QRadioButton("File Replay Mode (Local .txt files)")
        self.radio_file.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.radio_serial)
        mode_layout.addWidget(self.radio_modbus)
        mode_layout.addWidget(self.radio_file)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # ====== 串口参数表单 ======
        self.form_layout = QFormLayout()
        
        # 串口号下拉框
        self.com_combo = QComboBox()
        self.com_combo.setEditable(True)
        self.com_combo.addItems([
            "/dev/tty.usbserial-1410",
            "/dev/tty.usbmodem1101",
            "/dev/ttys001",
            "/dev/ttys002",
            "COM1",
            "COM2"
        ])
        
        # 波特率下拉框：默认 19200
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "115200"])
        self.baud_combo.setCurrentText("19200")
        
        self.form_layout.addRow("Com Port:", self.com_combo)
        self.form_layout.addRow("Baud Rate:", self.baud_combo)
        
        # ====== Modbus TCP 选项（仅在 Modbus 模式显示） ======
        self.modbus_tcp_check = QCheckBox("Use TCP (Local Test)")
        self.modbus_tcp_check.setChecked(False)
        self.modbus_tcp_check.toggled.connect(self.on_modbus_tcp_toggled)
        self.modbus_tcp_check.setVisible(False)  # 初始隐藏
        self.form_layout.addRow("", self.modbus_tcp_check)
        
        # TCP 主机地址输入（初始隐藏）
        self.tcp_host_input = QLineEdit("127.0.0.1")
        self.tcp_host_input.setPlaceholderText("e.g. 127.0.0.1")
        self.tcp_host_input.setVisible(False)
        self.form_layout.addRow("TCP Host:", self.tcp_host_input)
        
        # TCP 端口号输入（初始隐藏）
        self.tcp_port_input = QLineEdit("5020")
        self.tcp_port_input.setPlaceholderText("e.g. 5020")
        self.tcp_port_input.setVisible(False)
        self.form_layout.addRow("TCP Port:", self.tcp_port_input)
        
        main_layout.addLayout(self.form_layout)
        
        # ====== 提示标签 ======
        self.lbl_hint = QLabel("Mock String Mode: 连接虚拟串口模拟 4 通道声学数据")
        self.lbl_hint.setStyleSheet("color: #666; font-style: italic;")
        self.lbl_hint.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_hint)
        
        # ====== 保存按钮 ======
        self.save_btn = QPushButton("Save & Connect")
        self.save_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 8px;")
        self.save_btn.clicked.connect(self.accept)
        main_layout.addWidget(self.save_btn)
        
        # ====== 内部状态 ======
        self.file_replay_mode = False
        self.modbus_mode = False
        self.use_tcp = False
    
    def on_mode_changed(self):
        """
        当用户切换数据源模式时调用。
        - Mock String: 显示串口参数（串口号 + 波特率）
        - Modbus RTU: 显示串口 + TCP 选项
        - File Replay: 隐藏串口参数
        """
        self.file_replay_mode = False
        self.modbus_mode = False
        
        if self.radio_serial.isChecked():
            # ====== Mock String Mode ======
            self.form_layout.setEnabled(True)
            self.com_combo.setVisible(True)
            self.baud_combo.setVisible(True)
            self.modbus_tcp_check.setVisible(False)
            self.tcp_host_input.setVisible(False)
            self.tcp_port_input.setVisible(False)
            self.lbl_hint.setText("Mock String Mode: 连接虚拟串口模拟 4 通道声学数据")
            
        elif self.radio_modbus.isChecked():
            # ====== Modbus RTU Mode ======
            self.modbus_mode = True
            self.form_layout.setEnabled(True)
            self.com_combo.setVisible(True)
            self.baud_combo.setVisible(True)
            self.modbus_tcp_check.setVisible(True)
            # 根据 TCP 勾选状态决定是否显示主机/端口
            if self.modbus_tcp_check.isChecked():
                self.tcp_host_input.setVisible(True)
                self.tcp_port_input.setVisible(True)
                self.baud_combo.setVisible(False)
            else:
                self.tcp_host_input.setVisible(False)
                self.tcp_port_input.setVisible(False)
                self.baud_combo.setVisible(True)
            self.lbl_hint.setText(
                "Modbus RTU: Slave ID=126, 19200 8 E 1 | "
                "勾选 TCP 用于本地 Mock 服务器调试"
            )
            
        else:
            # ====== File Replay Mode ======
            self.file_replay_mode = True
            self.form_layout.setEnabled(False)
            self.com_combo.setVisible(False)
            self.baud_combo.setVisible(False)
            self.modbus_tcp_check.setVisible(False)
            self.tcp_host_input.setVisible(False)
            self.tcp_port_input.setVisible(False)
            self.lbl_hint.setText("File Replay: 选择包含 .txt 现场数据文件的文件夹")
    
    def on_modbus_tcp_toggled(self):
        """当用户切换 TCP 模式时，显示/隐藏 TCP 主机和端口输入框"""
        if self.modbus_tcp_check.isChecked():
            self.tcp_host_input.setVisible(True)
            self.tcp_port_input.setVisible(True)
            self.baud_combo.setVisible(False)  # TCP 模式不需要波特率
            self.lbl_hint.setText(
                "Modbus TCP: 连接本地 Mock 服务器 (127.0.0.1:5020)"
            )
        else:
            self.tcp_host_input.setVisible(False)
            self.tcp_port_input.setVisible(False)
            self.baud_combo.setVisible(True)
            self.lbl_hint.setText(
                "Modbus RTU: Slave ID=126, 19200 8 E 1"
            )
