"""
SandAlert PC - 零污染虚拟串口硬件模拟器（双引擎版）
==================================================
功能：在 macOS 上创建一个虚拟串口（伪终端），支持两种模拟模式：

  模式 1 - 现场历史回放（传统单探头定量评估）
    从指定的文件夹读取所有 .txt 文件，逐行发送逗号分隔的 4 通道数据。
    Ch.1 = 真实电压值, Ch.2~4 = 0.0
    目的：维持原有 IPS/出砂量功能，但 TDOA 阵列定位无法工作。

  模式 2 - TDOA 阵列空间定位（4 通道阵列模拟）
    完全基于数学模型实时生成 4 通道 UCA 阵列数据。
    每隔随机时间注入高频 Chirp 瞬态脉冲，带有物理精确的微秒级时延。
    目的：使 TDOA 雷达图在对应方位角稳定打点。

数据通信协议（兼容未来真实硬件）：
  f"{ch1},{ch2},{ch3},{ch4}\n"   （逗号分隔）

使用方法：
  1. 运行本模拟器，选择模式
  2. 根据提示在 Setup -> Comm Settings... 中连接虚拟串口
  3. 回到本终端按回车开始发送数据

作者：AI Assistant
日期：2026-05-17
"""

import os
import pty
import time
import glob
import sys
import random
import numpy as np
from core.tdoa_engine import TDOAEngine


# =============================================================================
#  引擎 A：现场历史数据回放（传统单探头）
# =============================================================================
def _engine_history_replay(data_dir: str, master_fd: int):
    """
    引擎 A：从指定目录读取 .txt 文件，循环发送逗号分隔的 4 通道数据。

    Ch.1 = 从文件读出的真实电压值
    Ch.2 = 0.0
    Ch.3 = 0.0
    Ch.4 = 0.0

    参数
    ----------
    data_dir : str
        包含 .txt 数据文件的文件夹路径。
    master_fd : int
        pty 主设备文件描述符。
    """
    search_pattern = os.path.join(data_dir, "*.txt")
    txt_files = sorted(glob.glob(search_pattern))

    if not txt_files:
        print(f"❌ 在 {data_dir} 中未找到任何 .txt 文件！")
        return

    print(f"📂 找到 {len(txt_files)} 个数据文件:")
    for f in txt_files:
        print(f"   - {os.path.basename(f)}")
    print()

    print(f"📡 开始向虚拟串口发送数据（共 {len(txt_files)} 个文件，循环发送）...")
    print("   按 Ctrl+C 可安全停止模拟器。\n")

    # 统计信息
    total_lines = 0
    file_index = 0

    try:
        while True:
            file_path = txt_files[file_index % len(txt_files)]
            file_name = os.path.basename(file_path)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 解析：原文件每行可能包含空格/Tab 分隔的多值
                    tokens = line.split()
                    for token in tokens:
                        try:
                            val = float(token)
                            # 格式：ch1,ch2,ch3,ch4
                            # Ch.1 = 真实电压, Ch.2~4 = 0.0
                            packet = f"{val},0.0,0.0,0.0\n"
                            os.write(master_fd, packet.encode('utf-8'))
                            time.sleep(0.001)
                            total_lines += 1
                        except ValueError:
                            # 跳过非数字 token（如文件头说明文字）
                            pass

            print(f"📄 [{file_name}] 发送完成  |  累计已发送 {total_lines} 行")
            file_index += 1

    except KeyboardInterrupt:
        print(f"\n🛑 模拟器已安全关闭。累计发送 {total_lines} 行数据。")
    except Exception as e:
        print(f"\n⚠️ 发送异常: {e}")


# =============================================================================
#  引擎 B：TDOA 阵列空间定位（4 通道阵列模拟）— 压缩时间块状版
# =============================================================================
def _engine_tdoa_array(master_fd: int):
    """
    引擎 B：基于数学模型的 4 通道 UCA 阵列信号实时生成（块状交替版）。

    生成策略（压缩时间，演示优化）：
    - Step A (极短底噪)：发送 500 行低幅值高斯白噪声
    - Step B (高能撞击)：生成 ~1500 行带物理时延的 Chirp 突刺数据块，峰值 ≈ 1.0V
    - 循环交替 Step A → Step B → Step A → Step B ...
    - 每块发送完毕后做一次批量 sleep，而非逐行 sleep

    参数
    ----------
    master_fd : int
        pty 主设备文件描述符。
    """
    print("\n📡 开始生成 4 通道阵列信号（压缩时间块状版）...")
    print("   按 Ctrl+C 可安全停止模拟器。\n")

    # ---- 初始化 TDOA 引擎获取几何参数 ----
    engine = TDOAEngine(fs=2_000_000, radius=0.05, c=1500.0)
    fs = engine.fs
    sensor_pos = engine.sensor_pos  # shape (4, 2)

    # ---- 物理参数 ----
    NOISE_STD = 0.002                 # 白噪声标准差（伏特）
    PULSE_AMPLITUDE = 1.2             # 脉冲幅度（保证峰值 ≈ 1.0 V）
    NOISE_BLOCK_LEN = 500             # Step A：底噪块长度
    IMPACT_BLOCK_LEN = 1500           # Step B：撞击块长度
    SEND_DELAY = 0.01                 # 每块发送后的延迟（秒）

    try:
        sample_idx = 0
        block_count = 0

        while True:
            # ============================================================
            #  Step A：发送极短底噪（仅 500 行，快速跳过）
            # ============================================================
            noise_block = np.random.randn(NOISE_BLOCK_LEN, 4).astype(np.float32) * NOISE_STD
            lines_a = []
            for row in noise_block:
                lines_a.append(
                    f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n"
                )
            for line in lines_a:
                os.write(master_fd, line.encode('utf-8'))
            sample_idx += NOISE_BLOCK_LEN

            # ============================================================
            #  Step B：生成并发送一个高能撞击事件
            # ============================================================
            # 1) 随机方位角
            #current_angle = random.uniform(0, 360)
            current_angle = 134.0 # 🎯 强制定角测试！改成 45 度！
            theta_rad = np.deg2rad(current_angle)
            source_dir = np.array([np.cos(theta_rad), np.sin(theta_rad)])

            # 2) 物理时延 tau_i = -p_i·u / c
            delays_true = -(sensor_pos @ source_dir) / 1500.0  # c = 1500 m/s
            delays_samples = np.round(delays_true * fs).astype(int)

            # 3) 构建高频 Chirp 突刺（200 kHz 正弦，无额外包络 = 尖锐突刺）
            t_axis = np.arange(IMPACT_BLOCK_LEN) / fs
            chirp_waveform = PULSE_AMPLITUDE * np.sin(
                2 * np.pi * 200e3 * t_axis
            )

            # 4) 构建 4 通道撞击块（底噪 + 带时延的 Chirp）
            impact_block = (
                np.random.randn(IMPACT_BLOCK_LEN, 4).astype(np.float32) * NOISE_STD
            )
            for ch_idx in range(4):
                delay = delays_samples[ch_idx]
                for i in range(IMPACT_BLOCK_LEN):
                    src_idx = i - delay
                    if 0 <= src_idx < IMPACT_BLOCK_LEN:
                        impact_block[i, ch_idx] += chirp_waveform[src_idx]

            # 5) 逐行发送撞击数据
            lines_b = []
            for row in impact_block:
                lines_b.append(
                    f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n"
                )
            for line in lines_b:
                os.write(master_fd, line.encode('utf-8'))
            sample_idx += IMPACT_BLOCK_LEN
            block_count += 1

            # ---- 打印撞击报告 ----
            peak_val = float(np.max(np.abs(impact_block)))
            print(f"💥 [{block_count}] 角度: {current_angle:.1f}°  "
                  f"| 时延: {list(delays_samples)} samples  "
                  f"| 峰值: {peak_val:.3f}V  "
                  f"| 累计行: {sample_idx}")

            # ---- 块间延迟（丝滑但不过快） ----
            time.sleep(SEND_DELAY)

    except KeyboardInterrupt:
        print(f"\n🛑 模拟器已安全关闭。累计发送 {sample_idx} 行数据。")
    except Exception as e:
        print(f"\n⚠️ 发送异常: {e}")


# =============================================================================
#  主入口：交互菜单 + pty 创建 + 双引擎分发
# =============================================================================
def start_mock_hardware(data_dir="."):
    """
    启动虚拟串口硬件模拟器（双引擎版）。

    流程：
    1. 打印交互菜单让用户选择模式
    2. 创建 pty 虚拟串口
    3. 输出串口信息供上位机连接
    4. 等待用户按回车确认
    5. 根据选定模式启动对应的数据引擎

    参数
    ----------
    data_dir : str
        模式 1 下 .txt 数据文件的文件夹路径（默认由 __main__ 传入）。
    """
    # ========== 步骤 1：交互菜单 ==========
    print("=" * 60)
    print("  SandAlert PC - 零污染硬件模拟器（双引擎版）")
    print("=" * 60)
    print()
    print("请选择硬件模拟模式：")
    print("  [1] 播放现场历史数据 (传统单探头定量评估)")
    print("  [2] 生成高频阵列信号 (4通道 TDOA 空间定位)")
    print()

    while True:
        try:
            choice = input("请输入数字 (1 or 2): ").strip()
            if choice in ("1", "2"):
                selected_mode = int(choice)
                break
            print("❌ 无效输入，请重新输入 1 或 2。")
        except (KeyboardInterrupt, EOFError):
            print("\n\n🛑 用户中断，退出。")
            return

    print()

    # ========== 步骤 2：创建 pty 虚拟串口 ==========
    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)

    print("=" * 60)
    print("🚀 [零污染硬件模拟器] 已启动！")
    print(f"👉 请在 SandAlert PC 的 Setup -> Comm Settings... 中：")
    print(f"   Com Port 设置为: {slave_name}")
    print(f"   Baud Rate 设置为: 19200")
    print(f"   数据格式: ch1,ch2,ch3,ch4  (逗号分隔)")
    print("=" * 60)
    print()

    # ========== 步骤 3：打印数据源信息 ==========
    if selected_mode == 1:
        print(f"📌 模式 1：播放现场历史数据")
        if not os.path.isdir(data_dir):
            print(f"❌ 数据路径不存在: {data_dir}")
            print(f"   请检查外置盘是否已挂载。")
            os.close(master)
            os.close(slave)
            return
        print(f"📂 数据路径: {data_dir}")
        print(f"   格式: ch1=真实电压, ch2~4=0.0")
    else:
        print(f"📌 模式 2：生成高频阵列信号 (4通道 TDOA 空间定位)")
        print(f"   格式: 4 通道 UCA (半径 0.05m, 声速 1500m/s)")
        print(f"   撞击注入: 随机角度, 每 0.5~1.5s 一次")

    print()

    # ========== 步骤 4：等待用户确认 ==========
    input("[等待操作] 请在软件中连接上述串口后，按回车键开始发送数据...\n")

    # ========== 步骤 5：启动对应引擎 ==========
    try:
        if selected_mode == 1:
            _engine_history_replay(data_dir, master)
        else:
            _engine_tdoa_array(master)
    finally:
        os.close(master)
        os.close(slave)
        print("🔌 虚拟串口已关闭。")


# =============================================================================
#  脚本入口
# =============================================================================
if __name__ == "__main__":
    # ====== 默认数据路径（仅用于模式 1） ======
    DEFAULT_DATA_DIR = ("/Volumes/LKS/中海油能源发展装备公司"
                        "海上油气管道含砂监测技术应用研究/结题资料/现场数据")

    # 如果命令行传入了文件夹路径，则使用它
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = DEFAULT_DATA_DIR
        print(f"📌 未指定数据路径，使用默认路径：{data_dir}")
        print(f"   如需自定义路径，请运行: python mock_hardware.py /你的/数据/路径\n")

    start_mock_hardware(data_dir)
