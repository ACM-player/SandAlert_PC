#!/usr/bin/env python3
"""
tdoa_estimator.py
=================
基于 TDOA + GCC-PHAT 的高频空间定位系统 (V2.0)

物理背景：工业管道声发射（AE）含砂监测。
核心策略：彻底放弃频域子空间分解（MUSIC/CSM），转向纯时域的到达时间差（TDOA）定位方案。

算法流程：
  1. 阵列信号建模：4 通道均匀圆阵 (UCA) 接收 300kHz 超高频宽带 Chirp 信号
  2. GCC-PHAT 时延估计：参考通道与其他通道进行广义互相关（带相位变换），
     提取互相关曲线全局峰值索引计算 Δt_ij
  3. 网格搜索 DOA 估计：0°~360° 扫描，最小化观测时延与理论时延的欧氏距离
  4. 可视化：代价函数倒数（似然谱）的极坐标图

作者：AI Assistant
日期：2026-05-16
"""

import numpy as np
from scipy.signal import chirp
from scipy.fft import fft, ifft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  全局参数配置
# ============================================================
FS = 2_000_000            # 采样率 2 MHz（奈奎斯特无压力，纯时域！）
DURATION = 0.005          # 信号时长 5 ms（超高频瞬态短脉冲）
N_SAMPLES = int(FS * DURATION)

M = 4                     # 传感器通道数
RADIUS = 0.05             # 阵列半径 5 cm
SOUND_SPEED = 1500.0      # 流体中声速 1500 m/s

TRUE_ANGLE_DEG = 60.0     # 真实撞击方位角（度）
SNR_DB = 10.0             # 信噪比 dB

# 高频 Chirp 参数（300kHz 超声发射频段）
F0 = 100e3                # 起始频率 100 kHz
F1 = 300e3                # 终止频率 300 kHz

# 网格搜索参数
ANGLE_STEP = 0.5          # 扫描步长（度）

# 参考通道（0-based index, 即 Ch.1）
REF_CHANNEL = 0

# 输出图片（使用脚本所在目录的绝对路径）
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_COST = os.path.join(_SCRIPT_DIR, "tdoa_cost_polar.png")


# ============================================================
#  第一步：恢复高频物理环境 —— 阵列信号生成
# ============================================================
def generate_array_data(
    m: int = M,
    radius: float = RADIUS,
    c: float = SOUND_SPEED,
    fs: int = FS,
    n_samples: int = N_SAMPLES,
    true_angle_deg: float = TRUE_ANGLE_DEG,
    snr_db: float = SNR_DB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成虚拟 4 通道均匀圆阵（UCA）接收的高频声发射数据。

    TDOA 基于包络互相关，天然免疫 180° 相位模糊，自信地用回 UCA。

    传感器布局：0°, 90°, 180°, 270°

    参数
    ----------
    m : int
        传感器数量。
    radius : float
        阵列半径（m）。
    c : float
        声速（m/s）。
    fs : int
        采样率（Hz）。
    n_samples : int
        采样点数。
    true_angle_deg : float
        真实撞击方位角（度），0 为 x 轴正方向。
    snr_db : float
        信噪比（dB）。

    返回
    -------
    X : np.ndarray, shape (m, n_samples)
        多通道接收信号（实数域）。
    t : np.ndarray, shape (n_samples,)
        时间轴。
    sensor_pos : np.ndarray, shape (m, 2)
        各传感器的 (x, y) 坐标。
    delays_true : np.ndarray, shape (m,)
        各传感器相对原点的真实物理时延（s）。
    source_signal : np.ndarray, shape (n_samples,)
        原始声源信号（无噪声）。
    """
    # 1. 传感器位置：均匀圆阵 (UCA) — 0°, 90°, 180°, 270°
    sensor_angles_deg = np.array([0.0, 90.0, 180.0, 270.0])
    sensor_angles_rad = np.deg2rad(sensor_angles_deg)
    sensor_pos = np.array([
        [radius * np.cos(phi), radius * np.sin(phi)]
        for phi in sensor_angles_rad
    ])  # shape (m, 2)

    # 2. 声源方向单位矢量（远场平面波假设）
    theta_true = np.deg2rad(true_angle_deg)
    source_dir = np.array([np.cos(theta_true), np.sin(theta_true)])  # (2,)

    # 3. 计算各传感器相对于原点的真实物理时延
    #    关键公式：tau_i = -(p_i · u) / c
    #    物理含义：距离声源越近的阵元，越早接收到信号（时延越小）
    #    负号原因：p_i · u 是投影到声源方向上的距离，
    #    远离声源侧投影为正 => 时延为正（晚到达），
    #    靠近声源侧投影为负 => 时延为负（早到达）。
    #    因此需要加负号将物理位置关系转化为正确的时序关系。
    delays_true = -(sensor_pos @ source_dir) / c  # shape (m,)
    print(f"  真实物理时延（相对原点）: {np.round(delays_true * 1e6, 2)} us")
    print(f"  最大时延差: {np.round((delays_true.max() - delays_true.min()) * 1e6, 2)} us")

    # 4. 生成声源信号 s(t) — 高频宽带 Chirp 瞬态冲击
    t = np.arange(n_samples) / fs
    t0 = 0.001  # 脉冲中心时刻 1 ms
    sigma = 0.0003  # 脉冲宽度 0.3 ms（适合 300kHz 高频瞬态）
    envelope = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

    # 高频 Chirp：100 kHz -> 300 kHz
    source_signal = envelope * chirp(t, f0=F0, t1=t0 + 2 * sigma,
                                     f1=F1, method='linear')
    source_signal[t < t0 - 3 * sigma] = 0.0
    source_signal[t > t0 + 3 * sigma] = 0.0

    # 5. 构造各通道接收信号（含时延）
    X = np.zeros((m, n_samples))
    for i in range(m):
        delay_samples = int(round(delays_true[i] * fs))
        # 使用循环移位实现正负时延（保留波形完整性）
        if delay_samples >= 0:
            X[i, delay_samples:] = source_signal[:n_samples - delay_samples]
        else:
            X[i, :delay_samples] = source_signal[-delay_samples:]

    # 6. 添加加性高斯白噪声（控制 SNR）
    signal_power = np.mean(source_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(m, n_samples)
    X_noisy = X + noise

    return X_noisy, t, sensor_pos, delays_true, source_signal


# ============================================================
#  第二步：GCC-PHAT 时延估计（核心算法）
# ============================================================
def gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
) -> tuple[float, np.ndarray]:
    """
    基于广义互相关-相位变换（GCC-PHAT）的两路信号时延估计。

    GCC-PHAT 的核心思想：
    - 在频域计算互功率谱 G_{12}(f) = conj(X1(f)) * X2(f)
    - 相位变换（PHAT）加权：G_{PHAT}(f) = G_{12}(f) / |G_{12}(f)|
      → 抹去幅度信息，仅保留相位信息
      → 等价于对互相关做白化处理，使互相关峰更尖锐
    - IFFT 回到时域得到广义互相关函数 R_{12}(τ)
    - 取 R_{12}(τ) 的全局峰值索引，得到时延 τ_est

    互相关定义：
      R_{12}(τ) = ∫ sig1(t) · sig2(t+τ) dt
      → 频域等价：G12(f) = conj(FFT(sig1)) · FFT(sig2)
      → 如果 sig2 比 sig1 晚 D 秒到达（sig2 向后移位），
         则 R_{12}(τ) 在 τ = +D 处出现峰值。
      → 所以返回正数时延表示 sig2 晚到。

    参数
    ----------
    sig1 : np.ndarray
        第一路信号（参考通道）。
    sig2 : np.ndarray
        第二路信号。
    fs : int
        采样率（Hz）。

    返回
    -------
    delay_sec : float
        估计的时延（秒），正数表示 sig2 比 sig1 晚到。
    cross_corr : np.ndarray
        广义互相关函数（经 PHAT 加权后）。
    """
    n = len(sig1)
    assert len(sig1) == len(sig2), "两路信号长度必须一致"

    # 1. FFT 到频域
    X1 = fft(sig1)
    X2 = fft(sig2)

    # 2. 互功率谱 G12(f) = conj(X1(f)) * X2(f)
    #    根据互相关定理: R_{12}(tau) <-> conj(FFT(sig1)) * FFT(sig2)
    #    若 sig2(t) = sig1(t - D) (sig2 比 sig1 晚到 D 秒),
    #    则 FFT(sig2) = FFT(sig1) * exp(-j*2*pi*f*D)
    #    G12 = conj(X1) * X2 = |X1|^2 * exp(-j*2*pi*f*D)
    #    IFFT(G12) 在 tau = +D 处有峰值
    G12 = np.conj(X1) * X2

    # 3. PHAT 加权：仅保留相位信息
    #    添加小常数 eps 防止除零
    G12_phat = G12 / (np.abs(G12) + 1e-30)

    # 4. IFFT 回到时域 → 广义互相关函数
    cross_corr = np.real(ifft(G12_phat))

    # 5. 循环移位修正：使 cross_corr 在 [-n/2, n/2] 范围内对称
    #    默认 IFFT 结果对应 [0, fs) 的时延区间
    #    我们将其重新排列为 [-n/2, fs - n/2) 对应负时延
    cross_corr = np.fft.fftshift(cross_corr)

    # 6. 全局峰值提取
    peak_idx = np.argmax(np.abs(cross_corr))
    # 将索引映射到时延（秒）
    # shift 后，索引范围 [0, n-1]，对应时延范围 [-n/2, n/2-1] 个采样点
    delay_samples = peak_idx - n // 2
    delay_sec = delay_samples / fs

    return delay_sec, cross_corr


def compute_tdoa_vector(
    X: np.ndarray,
    fs: int,
    ref_channel: int = REF_CHANNEL,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """
    对多通道信号，以参考通道为基准，计算所有通道的 TDOA 向量。

    参数
    ----------
    X : np.ndarray, shape (m, n_samples)
        多通道接收信号。
    fs : int
        采样率（Hz）。
    ref_channel : int
        参考通道索引。

    返回
    -------
    tdoa_obs : np.ndarray, shape (m-1,)
        观测到的相对时延向量 τ_obs = [Δt_{ref,1}, Δt_{ref,2}, ..., Δt_{ref,m-1}]。
        正数表示该通道比参考通道晚到。
    cross_corrs : list[np.ndarray]
        各通道与参考通道的 GCC-PHAT 互相关函数（用于调试/可视化）。
    delays_sec : list[float]
        各通道相对参考通道的时延估计值（秒）。
    """
    m = X.shape[0]
    ref_signal = X[ref_channel]

    tdoa_obs = np.zeros(m - 1)
    cross_corrs = []
    delays_sec = []

    for i in range(m):
        if i == ref_channel:
            delays_sec.append(0.0)
            continue

        delay_sec, xcorr = gcc_phat(ref_signal, X[i], fs)
        delays_sec.append(delay_sec)
        cross_corrs.append(xcorr)

        # 填充 tdoa 观测向量
        ch_idx = i - 1 if i > ref_channel else i
        tdoa_obs[ch_idx] = delay_sec
        delay_samples = int(round(delay_sec * fs))
        print(f"    Ch.{ref_channel+1} <-> Ch.{i+1}: "
              f"Δt = {delay_sec * 1e6:8.3f} us  "
              f"({delay_samples:4d} samples @ {fs/1e6:.1f} MHz)")

    return tdoa_obs, cross_corrs, delays_sec


# ============================================================
#  第三步：网格搜索法 DOA 估计
# ============================================================
def compute_theoretical_tdoa(
    theta_deg: float,
    sensor_pos: np.ndarray,
    c: float = SOUND_SPEED,
    ref_channel: int = REF_CHANNEL,
) -> np.ndarray:
    """
    计算某个候选角度 θ 下的理论相对时延向量 τ_theory(θ)。

    与数据生成使用完全一致的物理模型：
    tau_i(θ) = -(p_i · u(θ)) / c
    τ_theory(θ)_i = tau_i(θ) - tau_ref(θ)   （以参考通道为基准的相对时延）

    参数
    ----------
    theta_deg : float
        候选方位角（度）。
    sensor_pos : np.ndarray, shape (m, 2)
        各传感器坐标。
    c : float
        声速（m/s）。
    ref_channel : int
        参考通道索引。

    返回
    -------
    tau_theory : np.ndarray, shape (m-1,)
        相对于参考通道的理论时延向量（秒）。
    """
    theta_rad = np.deg2rad(theta_deg)
    source_dir = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # 各传感器相对于原点的时延
    delays = -(sensor_pos @ source_dir) / c  # shape (m,)

    # 以参考通道为基准的相对时延
    tau_theory = np.delete(delays, ref_channel) - delays[ref_channel]

    return tau_theory


def grid_search_doa(
    tdoa_obs: np.ndarray,
    sensor_pos: np.ndarray,
    c: float = SOUND_SPEED,
    angle_step: float = ANGLE_STEP,
    ref_channel: int = REF_CHANNEL,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    网格搜索方位角估计。

    对 0°~360° 每个候选角度 θ：
    1. 计算理论时延向量 τ_theory(θ)
    2. 代价函数 J(θ) = ||τ_obs - τ_theory(θ)||²  （欧氏距离平方）
    3. 使 J(θ) 最小的 θ 即为估计角度

    参数
    ----------
    tdoa_obs : np.ndarray, shape (m-1,)
        观测到的相对时延向量（GCC-PHAT 估计）。
    sensor_pos : np.ndarray, shape (m, 2)
        各传感器坐标。
    c : float
        声速（m/s）。
    angle_step : float
        扫描步长（度）。
    ref_channel : int
        参考通道索引。

    返回
    -------
    est_angle_deg : float
        估计的方位角（度）。
    angles_deg : np.ndarray
        扫描角度网格。
    cost_values : np.ndarray
        每个角度的代价函数值 J(θ)。
    """
    angles_deg = np.arange(0, 360, angle_step)
    N = len(angles_deg)
    cost_values = np.zeros(N)

    for i, theta in enumerate(angles_deg):
        tau_theory = compute_theoretical_tdoa(theta, sensor_pos, c, ref_channel)
        cost_values[i] = np.sum((tdoa_obs - tau_theory) ** 2)

    # 代价最小化 → 最佳估计
    best_idx = np.argmin(cost_values)
    est_angle_deg = angles_deg[best_idx]

    return est_angle_deg, angles_deg, cost_values


# ============================================================
#  第四步：可视化
# ============================================================
def plot_cost_function(
    angles_deg: np.ndarray,
    cost_values: np.ndarray,
    true_angle_deg: float,
    est_angle_deg: float,
    tdoa_obs: np.ndarray,
    delays_true_rel: np.ndarray,
):
    """
    生成代价函数的倒数（似然谱）极坐标图。

    将代价函数 J(θ) 转换为似然（谱值）：
    S(θ) = 1 / (J(θ) + ε)
    其中 ε 是一个极小的常数防止除零。

    同时也展示时延对比表格。
    """
    # ---------- 极坐标代价函数图 ----------
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},
                           figsize=(9, 9))

    angles_rad = np.deg2rad(angles_deg)

    # 代价函数 → 似然谱（倒数 + 归一化）
    likelihood = 1.0 / (cost_values + 1e-30)
    # 归一化到 [0, 1]
    l_min, l_max = likelihood.min(), likelihood.max()
    likelihood_norm = (likelihood - l_min) / (l_max - l_min + 1e-30)

    # 画出半透明填充和一个较细的曲线（更清晰的谱线）
    ax.plot(angles_rad, likelihood_norm, 'b-', linewidth=1.2,
            label='GCC-PHAT + Grid Search Likelihood')
    ax.fill(angles_rad, likelihood_norm, alpha=0.25, color='royalblue')

    # 标注真实角度
    true_rad = np.deg2rad(true_angle_deg)
    ax.plot(true_rad, 1.0, 'g*', markersize=20, markeredgecolor='white',
            markeredgewidth=1.2,
            label=f'True Angle = {true_angle_deg:.1f} deg')

    # 标注估计角度
    est_rad = np.deg2rad(est_angle_deg)
    ax.plot(est_rad, 1.0, 'rv', markersize=14, markeredgecolor='white',
            markeredgewidth=1.2,
            label=f'TDOA Est. = {est_angle_deg:.1f} deg')

    # 如果估计偏差较大，用弧线标注误差
    error_deg = abs(est_angle_deg - true_angle_deg)
    if error_deg > 180:
        error_deg = 360 - error_deg
    ax.set_title(
        'TDOA-GCC-PHAT Spatial Likelihood Spectrum\n'
        f'True: {true_angle_deg:.1f} deg  |  Est: {est_angle_deg:.1f} deg  |  '
        f'Error: {error_deg:.1f} deg',
        fontsize=14, pad=20)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=11)
    ax.grid(True, alpha=0.5)
    ax.set_theta_zero_location('E')  # 0 deg at right (math convention)
    ax.set_theta_direction(-1)  # CCW positive

    plt.tight_layout()
    plt.savefig(FIG_COST, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved polar cost function plot: {FIG_COST}")

    # ---------- 打印时延对比表 ----------
    print("\n" + "-" * 60)
    print("  TDOA Comparison Table (ref: Ch.1)")
    print("-" * 60)
    print(f"  {'Pair':<14} {'True DT (us)':<16} {'GCC-PHAT DT (us)':<18} {'Error (us)':<12}")
    print("  " + "-" * 58)
    for i in range(len(tdoa_obs)):
        true_val = delays_true_rel[i] * 1e6
        obs_val = tdoa_obs[i] * 1e6
        err = abs(true_val - obs_val)
        print(f"  {'Ch.1 <-> Ch.' + str(i + 2):<14} {true_val:<16.3f} {obs_val:<18.3f} {err:<12.3f}")
    print("-" * 60)

    # 打印角度估计结果
    print(f"\n  True azimuth: {true_angle_deg:.1f} deg")
    print(f"  TDOA estimate:  {est_angle_deg:.1f} deg")
    print(f"  Estimation error:   {error_deg:.1f} deg")


# ============================================================
#  主程序
# ============================================================
def main():
    print("=" * 65)
    print("  SandAlert TDOA V2 - GCC-PHAT based HF Source Localization")
    print("=" * 65)

    # ---------- Step 1: 生成高频阵列数据 ----------
    print("\n[Step 1] Generating virtual 4-ch UCA HF AE data...")
    np.random.seed(42)
    X, t, sensor_pos, delays_true, source_signal = generate_array_data()
    print(f"  Sample rate: {FS/1e6:.1f} MHz")
    print(f"  Duration: {DURATION*1000:.1f} ms,  {N_SAMPLES} samples")
    print(f"  Freq band: {F0/1e3:.0f} ~ {F1/1e3:.0f} kHz")
    print(f"  True azimuth: {TRUE_ANGLE_DEG:.1f} deg")
    print(f"  SNR: {SNR_DB:.0f} dB")
    print(f"  Array: UCA (0, 90, 180, 270) deg")
    print(f"  Ref channel: Ch.{REF_CHANNEL + 1}")

    # 计算以参考通道为基准的真实相对时延（用于对比验证）
    delays_true_rel = delays_true - delays_true[REF_CHANNEL]
    delays_true_rel = np.delete(delays_true_rel, REF_CHANNEL)

    print(f"\n  Per-channel delays (relative to ref):")
    for i in range(M):
        if i == REF_CHANNEL:
            print(f"    Ch.{i+1} (ref): DT = 0.000 us")
        else:
            ch_idx = i - 1 if i > REF_CHANNEL else i
            print(f"    Ch.{i+1}:        DT = {delays_true_rel[ch_idx] * 1e6:.3f} us")

    # ---------- Step 2: GCC-PHAT 时延估计 ----------
    print("\n[Step 2] GCC-PHAT delay estimation...")
    tdoa_obs, cross_corrs, delays_sec = compute_tdoa_vector(X, FS)

    # ---------- Step 3: 网格搜索 DOA 估计 ----------
    print("\n[Step 3] Grid search DOA estimation...")
    print(f"  Scan range: 0 ~ 360 deg, step = {ANGLE_STEP} deg")
    est_angle_deg, angles_deg, cost_values = grid_search_doa(
        tdoa_obs, sensor_pos, SOUND_SPEED, ANGLE_STEP)

    # ---------- Step 4: 可视化与输出 ----------
    print("\n[Step 4] Generating visualization...")
    plot_cost_function(angles_deg, cost_values, TRUE_ANGLE_DEG,
                       est_angle_deg, tdoa_obs, delays_true_rel)

    print("\n" + "=" * 65)
    print("  TDOA V2 Validation Complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
