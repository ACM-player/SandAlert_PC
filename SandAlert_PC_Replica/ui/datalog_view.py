"""
datalog_view.py — Data Logging "双核"数据解析面板

提供两个独立的导入入口：
  A) 加载标准 CSV 报表（Load CSV Report (File)）
  B) 加载原始 TXT 文件夹（Load Raw Data (Folder）

两种路径最终输出同一套统一数据格式：
  { x, ips, mass_rate, cumulative, total_mass, label }
交由底层 PlotWidget 无差别渲染。
"""

import os
import glob
import csv
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QFileDialog, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal


# ========================================================================
#  线程 B：TXT 文件夹解析器（后台线程，防止 UI 卡死）
# ========================================================================
class TxtFolderParserThread(QThread):
    """遍历文件夹内所有 .txt 文件，将每个文件的电压序列转为伪 IPS / 伪出砂量"""
    progress_update = Signal(int, int, str)   # 当前进度, 总数, 文件名
    data_ready = Signal(dict)                  # 统一数据字典
    error_msg = Signal(str)

    def __init__(self, data_dir, threshold=0.5, mass_factor=0.01):
        super().__init__()
        self.data_dir = data_dir
        self.threshold = threshold      # 电压阈值（超过此值计为一次"撞击"）
        self.mass_factor = mass_factor  # 绝对积分 → 质量转换系数
        self._is_running = False

    def run(self):
        self._is_running = True

        # ---- 搜集所有 .txt 文件并按文件名排序 ----
        search_pattern = os.path.join(self.data_dir, "*.txt")
        txt_files = sorted(glob.glob(search_pattern))

        if not txt_files:
            self.error_msg.emit("未在选定文件夹中找到任何 .txt 文件。")
            return

        x_list = []
        ips_list = []
        mass_list = []
        total_files = len(txt_files)

        for i, file_path in enumerate(txt_files):
            if not self._is_running:
                break

            file_name = os.path.basename(file_path)
            self.progress_update.emit(i + 1, total_files, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # ====== 读取数值 ======
                    tokens = f.read().split()
                    buffer = []
                    for token in tokens:
                        if not self._is_running:
                            break
                        try:
                            buffer.append(float(token))
                        except ValueError:
                            # 跳过非数值文本，防止崩溃
                            pass

                    if not buffer:
                        continue

                    voltage_array = np.array(buffer, dtype=np.float32)

                    # ====== 计算伪 IPS：超过阈值的采样点个数 ======
                    pseudo_ips = int(np.sum(voltage_array > self.threshold))

                    # ====== 计算伪出砂量：绝对积分能量 × 系数 ======
                    pseudo_mass = float(np.sum(np.abs(voltage_array)) * self.mass_factor)

                    x_list.append(i)
                    ips_list.append(pseudo_ips)
                    mass_list.append(pseudo_mass)

            except Exception as e:
                # 单个文件出错不中断整体流程
                print(f"[TXT Parser] 跳过文件 {file_name}: {e}")
                continue

        # ---- 组装统一数据格式 ----
        if not x_list:
            self.error_msg.emit("未能从任何 .txt 文件中解析出有效数值。")
            return

        x_arr = np.array(x_list, dtype=np.float64)
        ips_arr = np.array(ips_list, dtype=np.float64)
        mass_arr = np.array(mass_list, dtype=np.float64)
        cumulative_arr = np.cumsum(mass_arr)
        total_mass = float(cumulative_arr[-1]) if len(cumulative_arr) > 0 else 0.0

        result = {
            'x': x_arr,
            'ips': ips_arr,
            'mass_rate': mass_arr,
            'cumulative': cumulative_arr,
            'total_mass': total_mass,
            'label': os.path.basename(self.data_dir),
            'source': 'txt_folder',
        }
        self.data_ready.emit(result)

    def stop(self):
        self._is_running = False


# ========================================================================
#  DataLogWidget：双核解析面板
# ========================================================================
class DataLogWidget(QWidget):
    """
    历史数据解析与趋势展示面板

    提供两个导入按钮：
      - Load CSV Report (File)   → 解析标准 CSV 报表
      - Load Raw Data (Folder)    → 解析 TXT 文件夹中的原始电压数据

    两种入口最终输出统一格式的数据字典，交由渲染函数无差别绘图。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # ==================== 顶部控制栏 ====================
        self.control_layout = QHBoxLayout()

        # ---- 按钮 A：加载 CSV 报表 ----
        self.btn_load_csv = QPushButton("📄 Load CSV Report (File)")
        self.btn_load_csv.setStyleSheet(
            "background-color: #007bff; color: white; font-weight: bold; padding: 5px 10px;"
        )
        self.btn_load_csv.clicked.connect(self._on_load_csv)

        # ---- 按钮 B：加载 TXT 文件夹原始数据 ----
        self.btn_load_folder = QPushButton("📁 Load Raw Data (Folder)")
        self.btn_load_folder.setStyleSheet(
            "background-color: #6f42c1; color: white; font-weight: bold; padding: 5px 10px;"
        )
        self.btn_load_folder.clicked.connect(self._on_load_folder)

        # ---- 进度条 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)

        # ---- 信息标签 ----
        self.lbl_info = QLabel("Status: Ready. Load a CSV file or a folder of .txt files.")
        self.lbl_info.setStyleSheet("color: #555; font-style: italic;")

        # ---- 归集到顶栏 ----
        self.control_layout.addWidget(self.btn_load_csv)
        self.control_layout.addWidget(self.btn_load_folder)
        self.control_layout.addWidget(self.progress_bar)
        self.control_layout.addWidget(self.lbl_info)
        self.control_layout.addStretch()
        self.layout.addLayout(self.control_layout)

        # ==================== 双图纵向分割 ====================
        pg.setConfigOptions(antialias=True)
        self.plot_splitter = QSplitter(Qt.Vertical)

        # ---- 上图：砂率历史趋势（IPS / MassRate） ----
        self.rate_plot = pg.PlotWidget(title="Sand Production Rate Trend")
        self.rate_plot.setBackground('w')
        self.rate_plot.setLabel('left', 'IPS', units='count')
        self.rate_plot.setLabel('bottom', 'Time (Segment Index)')
        self.rate_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rate_plot.addLegend()

        # IPS 曲线（蓝色实线）
        self.ips_curve = self.rate_plot.plot(
            pen=pg.mkPen('#007bff', width=2),
            name='IPS',
        )
        # MassRate 曲线（橙色虚线）
        self.mass_curve = self.rate_plot.plot(
            pen=pg.mkPen('#ff7f0e', width=2, style=Qt.DashLine),
            name='MassRate',
        )

        # ---- 下图：累计出砂量 ----
        self.cum_plot = pg.PlotWidget(title="Cumulative Sand Mass")
        self.cum_plot.setBackground('w')
        self.cum_plot.setLabel('left', 'Cumulative Mass', units='g')
        self.cum_plot.setLabel('bottom', 'Time (Segment Index)')
        self.cum_plot.showGrid(x=True, y=True, alpha=0.3)

        # 累计曲线（橙色半透明填充）
        self.cum_curve = self.cum_plot.plot(
            pen=pg.mkPen('#ff7f0e', width=2),
            fillLevel=0,
            brush=(255, 127, 14, 50),
            name='CumulativeMass',
        )

        self.plot_splitter.addWidget(self.rate_plot)
        self.plot_splitter.addWidget(self.cum_plot)
        self.plot_splitter.setSizes([350, 350])
        self.layout.addWidget(self.plot_splitter)

        # ==================== 后台线程引用 ====================
        self.parser_thread = None

    # ----------------------------------------------------------------
    #  分支 A：解析标准 CSV 报表
    # ----------------------------------------------------------------
    def _on_load_csv(self):
        """按钮 A 回调：弹出文件选择器，解析 CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CSV 报表文件",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        self.lbl_info.setText(f"正在解析 CSV: {os.path.basename(file_path)}...")
        self.lbl_info.setStyleSheet("color: blue;")
        self.btn_load_csv.setEnabled(False)
        self.btn_load_folder.setEnabled(False)

        try:
            result = self._parse_csv_file(file_path)
            self._render_results(result)
            self.lbl_info.setText(
                f"CSV Loaded: {os.path.basename(file_path)} | "
                f"Total Records: {len(result['x'])} | "
                f"Total Mass: {result['total_mass']:.4f} g"
            )
            self.lbl_info.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            self.lbl_info.setText(f"CSV 解析失败: {str(e)}")
            self.lbl_info.setStyleSheet("color: red; font-weight: bold;")
        finally:
            self.btn_load_csv.setEnabled(True)
            self.btn_load_folder.setEnabled(True)

    def _parse_csv_file(self, file_path):
        """
        使用 csv.DictReader 智能解析 CSV 报表。

        支持标准表头格式：
          Timestamp, IPS, MassRate(g/s), Efficiency(%), Current(mA), CumulativeMass(g), TDOA_Angle(deg)

        通过模糊列名匹配提取：IPS、MassRate、CumulativeMass 三列。
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV 文件为空或无法读取表头。")

            col_names = [c.strip().lower() for c in reader.fieldnames]

            # ---- 模糊匹配列名 ----
            def find_col(keywords):
                """从 fieldnames 中找出包含任一关键字的第一列"""
                for raw_name, low_name in zip(reader.fieldnames, col_names):
                    for kw in keywords:
                        if kw in low_name:
                            return raw_name
                return None

            ips_col = find_col(['ips', 'impact rate', 'impact'])
            mass_col = find_col(['massrate', 'mass_rate', 'mass rate', 'mass'])
            cum_col = find_col(['cumulativemass', 'cumulative_mass', 'cumulative mass',
                                'cumulative', 'totalmass', 'total_mass', 'total mass'])

            if ips_col is None and mass_col is None and cum_col is None:
                raise ValueError(
                    f"无法在表头 {reader.fieldnames} 中找到 IPS/MassRate/CumulativeMass 列。\n"
                    f"请确保 CSV 包含标准表头。"
                )

            # ---- 逐行读取 ----
            x_data, ips_data, mass_data, cum_data = [], [], [], []
            for row_idx, row in enumerate(reader):
                try:
                    ips_val = float(row[ips_col]) if ips_col else 0.0
                    mass_val = float(row[mass_col]) if mass_col else 0.0
                    cum_val = float(row[cum_col]) if cum_col else 0.0
                except (ValueError, KeyError, TypeError):
                    continue

                x_data.append(row_idx)
                ips_data.append(ips_val)
                mass_data.append(mass_val)
                cum_data.append(cum_val)

            if not x_data:
                raise ValueError("CSV 中没有有效的数据行。")

        return {
            'x': np.array(x_data, dtype=np.float64),
            'ips': np.array(ips_data, dtype=np.float64),
            'mass_rate': np.array(mass_data, dtype=np.float64),
            'cumulative': np.array(cum_data, dtype=np.float64),
            'total_mass': float(cum_data[-1]) if cum_data else 0.0,
            'label': os.path.basename(file_path),
            'source': 'csv',
        }

    # ----------------------------------------------------------------
    #  分支 B：加载 TXT 文件夹
    # ----------------------------------------------------------------
    def _on_load_folder(self):
        """按钮 B 回调：弹出文件夹选择器，启动后台解析线程"""
        data_dir = QFileDialog.getExistingDirectory(
            self,
            "选择存放 .txt 原始电压数据的文件夹",
            "/",
        )
        if not data_dir:
            return

        self.lbl_info.setText(f"正在扫描文件夹: {os.path.basename(data_dir)}...")
        self.lbl_info.setStyleSheet("color: blue;")
        self.btn_load_csv.setEnabled(False)
        self.btn_load_folder.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # ---- 启动后台解析线程 ----
        # 使用默认阈值 0.5V 和质量系数 0.01（用户可后续通过 Parameters 面板校准）
        self.parser_thread = TxtFolderParserThread(
            data_dir, threshold=0.5, mass_factor=0.01
        )
        self.parser_thread.progress_update.connect(self._update_progress)
        self.parser_thread.data_ready.connect(self._on_txt_data_ready)
        self.parser_thread.error_msg.connect(self._on_txt_error)
        self.parser_thread.start()

    def _update_progress(self, current, total, file_name):
        """更新进度条"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_info.setText(f"Processing: {file_name} ({current}/{total})")

    def _on_txt_data_ready(self, result):
        """TXT 文件夹解析完成回调"""
        self._render_results(result)
        self.lbl_info.setText(
            f"Folder Loaded: {result['label']} | "
            f"Files: {len(result['x'])} | "
            f"Total Mass: {result['total_mass']:.4f} g"
        )
        self.lbl_info.setStyleSheet("color: green; font-weight: bold;")
        self.btn_load_csv.setEnabled(True)
        self.btn_load_folder.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_txt_error(self, msg):
        """TXT 文件夹解析出错回调"""
        self.lbl_info.setText(f"Error: {msg}")
        self.lbl_info.setStyleSheet("color: red; font-weight: bold;")
        self.btn_load_csv.setEnabled(True)
        self.btn_load_folder.setEnabled(True)
        self.progress_bar.setVisible(False)

    # ----------------------------------------------------------------
    #  统一渲染：两种数据来源共用同一绘图代码
    # ----------------------------------------------------------------
    def _render_results(self, data):
        """
        接受统一格式的数据字典，无差别渲染到两张图表上。

        数据字典格式:
        {
            'x': np.ndarray,           # 1D, 段索引或时间
            'ips': np.ndarray,         # 1D, 每段的 IPS
            'mass_rate': np.ndarray,   # 1D, 每段的出砂速率
            'cumulative': np.ndarray,  # 1D, 累积出砂量
            'total_mass': float,       # 最终总质量
            'label': str,              # 数据来源描述
            'source': 'csv' | 'txt_folder',
        }
        """
        x = data['x']
        ips = data['ips']
        mass_rate = data['mass_rate']
        cumulative = data['cumulative']

        # ---- 更新上图：砂率趋势 ----
        self.ips_curve.setData(x, ips)
        self.mass_curve.setData(x, mass_rate)

        # 自动调整 Y 轴范围
        self.rate_plot.autoRange()

        # ---- 更新下图：累计出砂量 ----
        self.cum_curve.setData(cumulative)
        self.cum_plot.autoRange()

    # ----------------------------------------------------------------
    #  安全关闭
    # ----------------------------------------------------------------
    def stop_parser(self):
        """由 MainWindow.closeEvent 调用，优雅终止后台解析线程"""
        if self.parser_thread is not None and self.parser_thread.isRunning():
            print("[Shutdown] 正在停止 TXT 文件夹解析线程...")
            self.parser_thread.stop()
            if not self.parser_thread.wait(2000):
                print("[Shutdown] 警告：TXT 解析线程未能在 2 秒内退出")
