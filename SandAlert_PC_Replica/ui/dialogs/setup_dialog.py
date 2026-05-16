"""
SandAlert PC - 通信设置对话框
==============================
功能：让用户选择数据源类型和通信参数。

支持两种数据源模式：
  1. 串口模式（默认）：连接真实硬件或虚拟串口
  2. 文件回放模式：从本地 .txt 文件回放数据（无硬件时的替代方案）

使用方法：
  点击菜单栏 Setup -> Comm Settings... 弹出此对话框。
  选择模式、串口号、波特率后，点击 "Save & Connect" 即可。
"""

from PySide6.QtWidgets import (QDialog, QFormLayout, QComboBox, QPushButton,
                               QLabel, QVBoxLayout, QGroupBox, QRadioButton)
from PySide6.QtCore import Qt

class SetupDialog(QDialog):
    """
    通信设置对话框：支持串口模式和文件回放模式两种数据源。
    
    属性:
        com_combo: 串口号下拉框
        baud_combo: 波特率下拉框
        file_replay_mode: 布尔值，True=文件回放模式，False=串口模式
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup - Communication Settings")
        self.setFixedSize(380, 250)
        
        # ====== 主布局 ======
        main_layout = QVBoxLayout(self)
        
        # ====== 数据源模式选择 ======
        mode_group = QGroupBox("Data Source Mode")
        mode_layout = QVBoxLayout()
        
        # 串口模式单选按钮（默认选中）
        self.radio_serial = QRadioButton("Serial Port (Hardware / Virtual)")
        self.radio_serial.setChecked(True)
        self.radio_serial.toggled.connect(self.on_mode_changed)
        
        # 文件回放模式单选按钮
        self.radio_file = QRadioButton("File Replay (Local .txt files)")
        self.radio_file.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.radio_serial)
        mode_layout.addWidget(self.radio_file)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # ====== 串口参数表单 ======
        self.form_layout = QFormLayout()
        
        # 串口号下拉框：预留 Mac 和 Windows 的常见串口格式
        self.com_combo = QComboBox()
        self.com_combo.setEditable(True)  # 允许用户手动输入自定义串口号
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
        main_layout.addLayout(self.form_layout)
        
        # ====== 提示标签 ======
        self.lbl_hint = QLabel("💡 提示：串口模式连接真实硬件或虚拟串口")
        self.lbl_hint.setStyleSheet("color: #666; font-style: italic;")
        self.lbl_hint.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_hint)
        
        # ====== 保存按钮 ======
        self.save_btn = QPushButton("Save & Connect")
        self.save_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 8px;")
        self.save_btn.clicked.connect(self.accept)
        main_layout.addWidget(self.save_btn)
        
        # ====== 内部状态 ======
        self.file_replay_mode = False  # 默认使用串口模式
    
    def on_mode_changed(self):
        """
        当用户切换数据源模式时调用。
        串口模式：显示串口参数表单
        文件回放模式：隐藏串口参数表单，提示用户后续会弹出文件夹选择器
        """
        if self.radio_serial.isChecked():
            # 串口模式：显示串口参数
            self.file_replay_mode = False
            self.form_layout.setEnabled(True)
            self.com_combo.setVisible(True)
            self.baud_combo.setVisible(True)
            self.lbl_hint.setText("💡 提示：串口模式连接真实硬件或虚拟串口")
        else:
            # 文件回放模式：隐藏串口参数
            self.file_replay_mode = True
            self.form_layout.setEnabled(False)
            self.com_combo.setVisible(False)
            self.baud_combo.setVisible(False)
            self.lbl_hint.setText("💡 提示：文件回放模式将弹出文件夹选择器")
