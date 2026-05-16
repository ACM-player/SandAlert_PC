#!/usr/bin/env python3
"""
array_signal_poc.py
====================
多通道阵列信号的空间滤波与方位估计 —— 纯算法概念验证（PoC）
[V3 - CSM 相干信号子空间法]

物理背景：油气管道含砂监测。砂粒撞击管壁产生宽带声发射（AE）信号。
核心目标：利用均匀圆阵（UCA）和 CSM (相干信号子空间法) + MVDR 算法，
          实现高精度宽带撞击方位估计与空间滤波。

算法流程：
  1. 阵列信号建模：虚拟 4 通道 UCA 接收信号（含时延 + 噪声）
  2. 带通滤波预处理：宽带 -> 窄带提取
  3. CSM 宽带空间谱扫描：
     a) CBF 粗扫获取预估角度 alpha_0
     b) 构造对角聚焦矩阵 T(f_k)，将所有频点聚焦到参考频率 f_0
     c) 合成相干协方差矩阵 R_C（多频点相干融合，天然满秩，不加对角加载）
     d) 单次 MUSIC 扫描获取精准 theta_est
  4. MVDR 最优权向量计算（含对角加载）
  5. 权向量合成 -> 还原纯净标量信号 y(t)
  6. 可视化验证

作者：AI Assistant
日期：2026-05-15
"""

import numpy as np
from scipy.signal import butter, sosfilt, chirp, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  全局参数配置
# ============================================================
FS = 1_000_000          # 采样率 1 MHz
DURATION = 0.015        # 信号时长 15 ms
N_SAMPLES = int(FS * DURATION)

M = 4                   # 传感器通道数
RADIUS = 0.05           # 阵列半径 5 cm
SOUND_SPEED = 1500.0    # 流体中声速 1500 m/s

TRUE_ANGLE_DEG = 45.0   # 真实撞击方位角（度）
SNR_DB = 10.0           # 信噪比 dB

# 带通滤波器参数（提取撞击信号主频段）
BP_LOW = 5e3            # 5 kHz
BP_HIGH = 15e3          # 15 kHz
FILTER_ORDER = 4

# CSM 扫描参数
ANGLE_STEP = 0.5        # 扫描步长（度）
N_FREQ_BINS = 50        # CSM 使用的频点数量（在通带内均匀选取）

# 对角加载系数（仅用于 MVDR，不用于 CSM 的协方差矩阵）
DIAG_LOAD_EPS = 1e-6

# 输出图片文件名
FIG_MUSIC = "music_spectrum.png"
FIG_COMPARE = "signal_comparison.png"


# ============================================================
#  辅助函数：Hilbert 变换（解析信号构造）
# ============================================================
def analytic_signal(x: np.ndarray) -> np.ndarray:
    """通过 scipy.signal.hilbert 构造实信号的解析信号（复数域）。"""
    return hilbert(x)


# ============================================================
#  第一步：阵列信号建模
# ============================================================
def generate_array_data(
    m: int = M,
    radius: float = RADIUS,
    c: float = SOUND_SPEED,
    fs: int = FS,
    n_samples: int = N_SAMPLES,
    true_angle_deg: float = TRUE_ANGLE_DEG,
    snr_db: float = SNR_DB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成虚拟 4 通道均匀圆阵（UCA）接收数据。

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
    sensor_angles : np.ndarray, shape (m,)
        各传感器的方位角（弧度）。
    """
    # 1. 传感器位置（非均匀圆阵 ACA，打破空间对称性消除 180° 模糊）
    sensor_angles_deg = np.array([0.0, 75.0, 145.0, 260.0])
    sensor_angles = np.deg2rad(sensor_angles_deg)
    sensor_pos = np.array([
        [radius * np.cos(phi), radius * np.sin(phi)]
        for phi in sensor_angles
    ])  # shape (m, 2)

    # 2. 声源方向单位矢量（远场平面波假设）
    theta_true = np.deg2rad(true_angle_deg)
    source_dir = np.array([np.cos(theta_true), np.sin(theta_true)])  # (2,)

    # 3. 计算各传感器相对于原点的时延（物理修正：添加负号！）
    #    距离声源越近的阵元，应该越早接收到信号（时延越小）
    #    tau_i = -(p_i . u) / c
    delays = -(sensor_pos @ source_dir) / c  # shape (m,)
    # 极其关键：为了在下方 X 矩阵赋值时不出现数组负索引导致时序环形错位，
    # 必须将整个阵列的相对时延平移为非负数。
    # （整体的时间平移等价于信号晚发出一会儿，不影响阵元间的相对相位差和方位估计）
    delays = delays - np.min(delays)

    # 4. 生成声源信号 s(t) -- 瞬态宽带 AE 脉冲
    t = np.arange(n_samples) / fs
    t0 = 0.005  # 脉冲中心时刻（延后以适应延长的 DURATION）
    sigma = 0.001  # 脉冲宽度（低频包络更宽）
    envelope = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    # 线性调频：从 5 kHz 扫到 15 kHz
    source_signal = envelope * chirp(t, f0=5e3, t1=t0 + 3 * sigma,
                                     f1=15e3, method='linear')
    source_signal[t < t0 - 3 * sigma] = 0.0
    source_signal[t > t0 + 3 * sigma] = 0.0

    # 5. 构造各通道接收信号（含时延）
    X = np.zeros((m, n_samples))
    for i in range(m):
        delay_samples = int(round(delays[i] * fs))
        if delay_samples >= 0:
            X[i, delay_samples:] = source_signal[:n_samples - delay_samples]
        else:
            X[i, :delay_samples] = source_signal[-delay_samples:]

    # 6. 添加加性高斯白噪声（控制 SNR）
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(m, n_samples)
    X += noise

    return X, t, sensor_angles


# ============================================================
#  预处理：带通滤波器（宽带 -> 窄带提取）
# ============================================================
def design_bandpass_filter(
    low: float = BP_LOW,
    high: float = BP_HIGH,
    fs: int = FS,
    order: int = FILTER_ORDER,
):
    """设计带通滤波器（二阶节形式）。"""
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sos


def apply_bandpass_filter(X: np.ndarray, sos) -> np.ndarray:
    """对多通道信号逐通道进行带通滤波。"""
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_filtered[i] = sosfilt(sos, X[i])
    return X_filtered


# ============================================================
#  第二步：CSM 宽带空间谱扫描（核心重构）
# ============================================================
def csm_scan(
    X: np.ndarray,
    sensor_angles: np.ndarray,
    radius: float = RADIUS,
    c: float = SOUND_SPEED,
    fs: int = FS,
    angle_step: float = ANGLE_STEP,
    n_freq_bins: int = N_FREQ_BINS,
    n_sources: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    使用 CSM (相干信号子空间法) 在 0~360 范围内扫描宽带空间功率谱。

    核心思想：
      1. FFT 到频域，在通带内选取 K 个有效频点
      2. CBF 粗扫获取预估角度 alpha_0
      3. 构造对角聚焦矩阵 T(f_k)，将所有频点聚焦到参考频率 f_0
      4. 合成相干协方差矩阵 R_C（多频点相干融合，天然满秩）
      5. 对 R_C 做一次 eigh，用 a(f_0, theta) 做 MUSIC 扫描

    参数
    ----------
    X : np.ndarray, shape (m, n_samples)
        滤波后的多通道信号。
    sensor_angles : np.ndarray, shape (m,)
        各传感器的方位角（弧度）。
    radius : float
        阵列半径（m）。
    c : float
        声速（m/s）。
    fs : int
        采样率（Hz）。
    angle_step : float
        扫描步长（度）。
    n_freq_bins : int
        在通带内选取的频点数量。
    n_sources : int
        信号源数量。

    返回
    -------
    angles_deg : np.ndarray
        扫描角度网格（度）。
    spectrum : np.ndarray
        CSM 宽带伪谱值。
    est_angle_deg : float
        估计的撞击方位角（度）。
    """
    m = X.shape[0]
    n_samples = X.shape[1]

    # ---- 1. FFT 将信号转换到频域 ----
    X_freq = np.fft.fft(X, axis=1)  # shape (m, n_samples)
    freqs = np.fft.fftfreq(n_samples, 1 / fs)

    # ---- 2. 在通带内选取有效频点 ----
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    X_freq_pos = X_freq[:, pos_mask]  # shape (m, n_pos)

    band_mask = (freqs_pos >= BP_LOW) & (freqs_pos <= BP_HIGH)
    freq_band = freqs_pos[band_mask]
    X_freq_band = X_freq_pos[:, band_mask]  # shape (m, n_band)

    n_band = len(freq_band)
    if n_band < n_freq_bins:
        selected_indices = np.arange(n_band)
    else:
        selected_indices = np.linspace(0, n_band - 1, n_freq_bins,
                                       dtype=int)

    selected_freqs = freq_band[selected_indices]  # shape (K,)
    X_selected = X_freq_band[:, selected_indices]  # shape (m, K)

    K = len(selected_freqs)
    print(f"  CSM: 通带内 {n_band} 个频点, 选取 {K} 个用于相干聚焦")

    # ---- 3. 参考频率（带通中心频率） ----
    f0 = (BP_LOW + BP_HIGH) / 2.0
    k0 = 2 * np.pi * f0 / c  # 参考波数

    # ---- 4. CBF 粗扫获取预估角度 alpha_0 ----
    # 使用常规波束形成（延迟求和）快速扫描
    angles_coarse = np.arange(0, 360, 1.0)  # 1 度步长粗扫
    theta_coarse = np.deg2rad(angles_coarse)
    cos_diff_coarse = np.cos(theta_coarse[:, np.newaxis] -
                             sensor_angles[np.newaxis, :])

    # 对每个粗扫角度，计算所有频点的 CBF 输出功率
    cbf_power = np.zeros(len(angles_coarse))
    for k_idx in range(K):
        kk = 2 * np.pi * selected_freqs[k_idx] / c
        A_coarse = np.exp(1j * kk * radius * cos_diff_coarse) / np.sqrt(m)
        x_k = X_selected[:, k_idx:k_idx + 1]  # (m, 1)
        # CBF 输出: y = a^H @ x, 功率 = |y|^2
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            y_cbf = A_coarse.conj() @ x_k  # (N_coarse, 1)
        y_cbf = np.nan_to_num(y_cbf, nan=0.0, posinf=0.0, neginf=0.0)
        cbf_power += np.abs(y_cbf.flatten()) ** 2

    # ---- 4b. CBF 粗扫取全局峰值作为预估角度 alpha_0 ----
    # 非均匀圆阵（ACA）已打破空间对称性，180° 模糊自然消除，
    # 直接取全局最大峰值即可。
    alpha0_deg = angles_coarse[np.argmax(cbf_power)]
    alpha0 = np.deg2rad(alpha0_deg)
    print(f"  CBF coarse estimate: {alpha0_deg:.1f} deg")

    # ---- 5. 构造聚焦矩阵并合成相干协方差矩阵 R_C ----
    # 对角聚焦矩阵 T(f_k):
    #   T_{m,m}(f_k) = exp(+j * 2*pi * (f0 - f_k) * r/c * cos(alpha0 - gamma_m))
    # 其中 gamma_m 是第 m 个传感器的方位角
    #
    # 聚焦后: Y(f_k) = T(f_k) @ X(f_k)
    # 相干协方差: R_C = (1/K) * sum_k Y(f_k) @ Y^H(f_k)

    R_C = np.zeros((m, m), dtype=complex)

    for k_idx in range(K):
        fk = selected_freqs[k_idx]
        kk = 2 * np.pi * fk / c

        # 聚焦矩阵对角线元素 (物理修正：正向相位补偿)
        # exp(+j * (k0 - kk) * r * cos(alpha0 - gamma_m))
        # = exp(+j * 2*pi * (f0 - fk) * r/c * cos(alpha0 - gamma_m))
        phase_diff = (k0 - kk) * radius * np.cos(alpha0 - sensor_angles)
        T_diag = np.exp(1j * phase_diff)  # shape (m,)

        # 聚焦: Y(f_k) = T(f_k) @ X(f_k)
        # T 是对角矩阵，所以 Y_m = T_{mm} * X_m
        Y_k = T_diag[:, np.newaxis] * X_selected[:, k_idx:k_idx + 1]

        # 累加相干协方差
        R_C += Y_k @ Y_k.conj().T

    R_C /= K  # 相干平均

    # ---- 6. 对 R_C 做特征值分解（不加对角加载！） ----
    eigenvalues, eigenvectors = np.linalg.eigh(R_C)
    En = eigenvectors[:, :m - n_sources]  # shape (m, m-n_sources)

    # ---- 7. 用参考频率的 steering vector 做单次 MUSIC 扫描 ----
    angles_deg = np.arange(0, 360, angle_step)
    N_theta = len(angles_deg)
    theta_rad = np.deg2rad(angles_deg)

    cos_diff = np.cos(theta_rad[:, np.newaxis] -
                      sensor_angles[np.newaxis, :])
    A0 = np.exp(1j * k0 * radius * cos_diff) / np.sqrt(m)

    # 严格使用共轭转置 a^H 进行子空间投影
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        A0_En = A0.conj() @ En  # shape (N_theta, m-n_sources)
    A0_En = np.nan_to_num(A0_En, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sum(np.abs(A0_En) ** 2, axis=1)  # shape (N_theta,)
    spectrum = 1.0 / (denom + 1e-30)

    # ---- 8. 找峰值（全局最大值） ----
    # 非均匀圆阵（ACA）已打破空间对称性，180° 镜像假峰自然消失，
    # 直接取全局最大峰值作为最终方位角估计。
    peak_idx = np.argmax(spectrum)
    est_angle_deg = angles_deg[peak_idx]

    return angles_deg, spectrum, est_angle_deg


# ============================================================
#  第三步：MVDR 最优权向量计算
# ============================================================
def mvdr_beamformer(
    X: np.ndarray,
    sensor_angles: np.ndarray,
    est_angle_deg: float,
    radius: float = RADIUS,
    c: float = SOUND_SPEED,
    diag_load_eps: float = DIAG_LOAD_EPS,
) -> np.ndarray:
    """
    根据估计的撞击方位角，计算 MVDR 最优权向量。
    （使用信号中心频率构造 steering vector）
    """
    m = X.shape[0]

    # 1. 构造解析信号计算协方差矩阵
    X_hilbert = np.zeros_like(X, dtype=complex)
    for i in range(m):
        X_hilbert[i] = analytic_signal(X[i])

    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        R = (X_hilbert @ X_hilbert.conj().T) / X.shape[1]
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. 对角加载（防止矩阵奇异）
    R_load = R + diag_load_eps * np.trace(R) / m * np.eye(m)

    # 3. 构造 steering vector a(theta_est) -- 使用中心频率
    fc = (BP_LOW + BP_HIGH) / 2.0
    wavelength = c / fc
    k = 2 * np.pi / wavelength

    theta_est = np.deg2rad(est_angle_deg)
    a = np.exp(1j * k * radius * np.cos(theta_est - sensor_angles))
    a = a / np.sqrt(m)

    # 4. MVDR 最优权向量
    R_inv = np.linalg.inv(R_load)
    numerator = R_inv @ a
    denominator = a.conj() @ numerator
    w_opt = numerator / denominator

    return w_opt


# ============================================================
#  第四步：权向量合成与信号还原
# ============================================================
def spatial_filtering(
    X: np.ndarray,
    w_opt: np.ndarray,
) -> np.ndarray:
    """
    将 MVDR 最优权向量作用于多通道信号，合成纯净标量信号。
    y(t) = w_opt^H * X(t)
    """
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        y = w_opt.conj() @ X
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.real(y)


# ============================================================
#  可视化验证
# ============================================================
def plot_results(
    angles_deg: np.ndarray,
    spectrum: np.ndarray,
    true_angle_deg: float,
    est_angle_deg: float,
    t: np.ndarray,
    X_raw: np.ndarray,
    y_mvdr: np.ndarray,
    fs: int = FS,
):
    """
    生成两张验证图并保存为本地文件。

    图1：CSM 宽带空间极坐标谱图
    图2：原始信号 vs MVDR 合成信号 时域对比图
    """
    # ---------- 图1：CSM 空间极坐标谱图 ----------
    fig1, ax1 = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},
                              figsize=(8, 8))

    angles_rad = np.deg2rad(angles_deg)
    # 归一化谱到 [0, 1]
    s_min, s_max = spectrum.min(), spectrum.max()
    spectrum_norm = (spectrum - s_min) / (s_max - s_min + 1e-30)

    ax1.plot(angles_rad, spectrum_norm, 'b-', linewidth=1.5,
             label='CSM Broadband Pseudo-spectrum')
    ax1.fill(angles_rad, spectrum_norm, alpha=0.3, color='blue')

    # 标注真实角度
    true_rad = np.deg2rad(true_angle_deg)
    ax1.plot(true_rad, 1.0, 'g*', markersize=18,
             label=f'True Angle = {true_angle_deg:.1f} deg')

    # 标注估计角度
    est_rad = np.deg2rad(est_angle_deg)
    ax1.plot(est_rad, 1.0, 'rv', markersize=12,
             label=f'CSM Est. = {est_angle_deg:.1f} deg')

    ax1.set_title('CSM Broadband Spatial Spectrum (Polar)',
                  fontsize=14, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_MUSIC, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"[OK] Saved CSM spectrum: {FIG_MUSIC}")

    # ---------- 图2：时域对比图 ----------
    fig2, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 子图1：原始 4 通道信号
    for i in range(M):
        axes[0].plot(t * 1e3, X_raw[i], linewidth=0.8, alpha=0.8,
                     label=f'Ch.{i+1}')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Raw 4-Channel UCA Signals', fontsize=12)
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 子图2：MVDR 合成信号
    axes[1].plot(t * 1e3, y_mvdr, 'r-', linewidth=1.2,
                 label='MVDR Beamformed')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('MVDR Spatial Filtering Output', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 子图3：对比叠加（通道1 vs MVDR）
    ch1_power = np.mean(X_raw[0] ** 2)
    mvdr_power = np.mean(y_mvdr ** 2)
    scale = np.sqrt(ch1_power / (mvdr_power + 1e-30))
    y_mvdr_scaled = y_mvdr * scale

    axes[2].plot(t * 1e3, X_raw[0], 'b-', linewidth=0.8, alpha=0.7,
                 label='Ch.1 (Raw)')
    axes[2].plot(t * 1e3, y_mvdr_scaled, 'r-', linewidth=1.2, alpha=0.9,
                 label='MVDR (Energy Aligned)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Ch.1 vs MVDR Output Comparison', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_COMPARE, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[OK] Saved signal comparison: {FIG_COMPARE}")


# ============================================================
#  主程序
# ============================================================
def main():
    print("=" * 60)
    print("  Broadband Array Spatial Filtering & DOA PoC [CSM v3]")
    print("=" * 60)

    # ---------- Step 1: 生成虚拟阵列数据 ----------
    print("\n[Step 1] Generating virtual 4-channel UCA data...")
    np.random.seed(42)
    X_raw, t, sensor_angles = generate_array_data()
    print(f"  Duration: {DURATION*1000:.1f} ms,  "
          f"Sample Rate: {FS/1e3:.0f} kHz")
    print(f"  Data shape: {X_raw.shape}")
    print(f"  True impact angle: {TRUE_ANGLE_DEG} deg")
    print(f"  SNR: {SNR_DB} dB")

    # ---------- 预处理：带通滤波 ----------
    print("\n[Step 2] Bandpass filtering ({:.0f}-{:.0f} kHz)...".format(
        BP_LOW/1e3, BP_HIGH/1e3))
    sos = design_bandpass_filter()
    X_filt = apply_bandpass_filter(X_raw, sos)
    print("  Filtering done")

    # ---------- Step 2: CSM 宽带空间谱扫描 ----------
    print("\n[Step 3] CSM broadband spatial spectrum scanning...")
    angles_deg, spectrum, est_angle_deg = csm_scan(X_filt, sensor_angles)
    print(f"  Estimated angle: {est_angle_deg:.1f} deg")
    print(f"  Estimation error: "
          f"{abs(est_angle_deg - TRUE_ANGLE_DEG):.1f} deg")

    # ---------- Step 3: MVDR 最优权向量 ----------
    print("\n[Step 4] Computing MVDR optimal weight vector "
          "(with diagonal loading)...")
    w_opt = mvdr_beamformer(X_filt, sensor_angles, est_angle_deg)
    print(f"  Weight vector shape: {w_opt.shape}")
    print(f"  Weights (real): {np.round(np.real(w_opt), 4)}")
    print(f"  Weights (imag): {np.round(np.imag(w_opt), 4)}")

    # ---------- Step 4: 空间滤波合成 ----------
    print("\n[Step 5] Spatial filtering & signal reconstruction...")
    y_mvdr = spatial_filtering(X_filt, w_opt)
    print(f"  Output signal length: {len(y_mvdr)} samples")

    # ---------- 计算信噪比改善 ----------
    ch1_power_before = np.mean(X_filt[0] ** 2)
    mvdr_power = np.mean(y_mvdr ** 2)
    noise_region = slice(-500, None)
    noise_power_ch1 = np.mean(X_filt[0, noise_region] ** 2)
    noise_power_mvdr = np.mean(y_mvdr[noise_region] ** 2)

    snr_before = 10 * np.log10(
        (ch1_power_before - noise_power_ch1) / (noise_power_ch1 + 1e-30))
    snr_after = 10 * np.log10(
        (mvdr_power - noise_power_mvdr) / (noise_power_mvdr + 1e-30))
    print(f"\n  SNR improvement: {snr_before:.1f} dB -> "
          f"{snr_after:.1f} dB "
          f"(gain: {snr_after - snr_before:.1f} dB)")

    # ---------- 可视化 ----------
    print("\n[Step 6] Generating visualization figures...")
    plot_results(angles_deg, spectrum, TRUE_ANGLE_DEG, est_angle_deg,
                 t, X_raw, y_mvdr)

    print("\n" + "=" * 60)
    print("  PoC Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
