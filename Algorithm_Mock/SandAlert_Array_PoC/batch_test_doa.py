#!/usr/bin/env python3
"""
batch_test_doa.py
==================
CSM 宽带 DOA 算法鲁棒性验证 —— Monte Carlo 批量测试

功能：
  1. 随机生成 N 个 0~360° 的真实撞击角度
  2. 对每个角度运行 CSM 宽带 DOA 估计（CBF 粗扫 + 聚焦矩阵 + MUSIC 精扫）
  3. 统计 MAE / Max Error / RMSE
  4. 生成真实 vs 估计角度散点图

作者：AI Assistant
日期：2026-05-15
"""

import numpy as np
from scipy.signal import butter, sosfilt, chirp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  全局参数配置（与 array_signal_poc.py 保持一致）
# ============================================================
FS = 1_000_000
DURATION = 0.015
N_SAMPLES = int(FS * DURATION)

M = 4
RADIUS = 0.05
SOUND_SPEED = 1500.0

SNR_DB = 10.0

BP_LOW = 5e3
BP_HIGH = 15e3
FILTER_ORDER = 4

ANGLE_STEP = 0.5
N_FREQ_BINS = 50

# 传感器位置（非均匀圆阵 ACA）
SENSOR_ANGLES_DEG = np.array([0.0, 75.0, 145.0, 260.0])
SENSOR_ANGLES = np.deg2rad(SENSOR_ANGLES_DEG)
SENSOR_POS = np.array([
    [RADIUS * np.cos(phi), RADIUS * np.sin(phi)]
    for phi in SENSOR_ANGLES
])

# 输出图片文件名
FIG_SCATTER = "doa_scatter.png"


# ============================================================
#  核心函数：单次 DOA 估计
# ============================================================
def evaluate_single_doa(true_angle_deg: float, snr_db: float = SNR_DB) -> float:
    """
    对单个真实角度运行完整的 CSM 宽带 DOA 估计流程。

    参数
    ----------
    true_angle_deg : float
        真实撞击方位角（度）。
    snr_db : float
        信噪比（dB）。

    返回
    -------
    est_angle_deg : float
        CSM 算法估计的方位角（度）。
    """
    # ---- Step 1: 生成虚拟阵列数据 ----
    theta_true = np.deg2rad(true_angle_deg)
    source_dir = np.array([np.cos(theta_true), np.sin(theta_true)])

    # 时延（物理修正：负号 + 非负平移）
    delays = -(SENSOR_POS @ source_dir) / SOUND_SPEED
    delays = delays - np.min(delays)

    # 声源信号
    t = np.arange(N_SAMPLES) / FS
    t0 = 0.005
    sigma = 0.001
    envelope = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    source_signal = envelope * chirp(t, f0=5e3, t1=t0 + 3 * sigma,
                                     f1=15e3, method='linear')
    source_signal[t < t0 - 3 * sigma] = 0.0
    source_signal[t > t0 + 3 * sigma] = 0.0

    # 多通道接收
    X = np.zeros((M, N_SAMPLES))
    for i in range(M):
        d = int(round(delays[i] * FS))
        if d >= 0:
            X[i, d:] = source_signal[:N_SAMPLES - d]
        else:
            X[i, :d] = source_signal[-d:]

    # 加噪声
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    X += np.sqrt(noise_power) * np.random.randn(M, N_SAMPLES)

    # ---- Step 2: 带通滤波 ----
    sos = butter(FILTER_ORDER, [BP_LOW, BP_HIGH], btype='band', fs=FS, output='sos')
    X_filt = np.zeros_like(X)
    for i in range(M):
        with np.errstate(invalid='ignore', divide='ignore'):
            X_filt[i] = sosfilt(sos, X[i])
    # 清理滤波可能产生的 inf/nan
    X_filt = np.nan_to_num(X_filt, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Step 3: CSM 宽带空间谱扫描 ----
    # 3a. FFT
    X_freq = np.fft.fft(X_filt, axis=1)
    freqs = np.fft.fftfreq(N_SAMPLES, 1 / FS)

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    X_freq_pos = X_freq[:, pos_mask]

    band_mask = (freqs_pos >= BP_LOW) & (freqs_pos <= BP_HIGH)
    freq_band = freqs_pos[band_mask]
    X_freq_band = X_freq_pos[:, band_mask]

    n_band = len(freq_band)
    if n_band < N_FREQ_BINS:
        selected_indices = np.arange(n_band)
    else:
        selected_indices = np.linspace(0, n_band - 1, N_FREQ_BINS, dtype=int)

    selected_freqs = freq_band[selected_indices]
    X_selected = X_freq_band[:, selected_indices]
    K = len(selected_freqs)

    # 3b. 参考频率
    f0 = (BP_LOW + BP_HIGH) / 2.0
    k0 = 2 * np.pi * f0 / SOUND_SPEED

    # 3c. CBF 粗扫
    angles_coarse = np.arange(0, 360, 1.0)
    theta_coarse = np.deg2rad(angles_coarse)
    cos_diff_coarse = np.cos(theta_coarse[:, np.newaxis] -
                             SENSOR_ANGLES[np.newaxis, :])

    cbf_power = np.zeros(len(angles_coarse))
    for k_idx in range(K):
        kk = 2 * np.pi * selected_freqs[k_idx] / SOUND_SPEED
        A_coarse = np.exp(1j * kk * RADIUS * cos_diff_coarse) / np.sqrt(M)
        x_k = X_selected[:, k_idx:k_idx + 1]
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            y_cbf = A_coarse.conj() @ x_k
        y_cbf = np.nan_to_num(y_cbf, nan=0.0, posinf=0.0, neginf=0.0)
        cbf_power += np.abs(y_cbf.flatten()) ** 2

    alpha0_deg = angles_coarse[np.argmax(cbf_power)]
    alpha0 = np.deg2rad(alpha0_deg)

    # 3d. 构造聚焦矩阵 + 合成相干协方差 R_C
    R_C = np.zeros((M, M), dtype=complex)
    for k_idx in range(K):
        fk = selected_freqs[k_idx]
        kk = 2 * np.pi * fk / SOUND_SPEED
        phase_diff = (k0 - kk) * RADIUS * np.cos(alpha0 - SENSOR_ANGLES)
        T_diag = np.exp(1j * phase_diff)
        Y_k = T_diag[:, np.newaxis] * X_selected[:, k_idx:k_idx + 1]
        R_C += Y_k @ Y_k.conj().T
    R_C /= K

    # 3e. 特征值分解 + MUSIC 扫描
    eigenvalues, eigenvectors = np.linalg.eigh(R_C)
    En = eigenvectors[:, :M - 1]  # n_sources = 1

    angles_deg = np.arange(0, 360, ANGLE_STEP)
    theta_rad = np.deg2rad(angles_deg)
    cos_diff = np.cos(theta_rad[:, np.newaxis] -
                      SENSOR_ANGLES[np.newaxis, :])
    A0 = np.exp(1j * k0 * RADIUS * cos_diff) / np.sqrt(M)

    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        A0_En = A0.conj() @ En
    A0_En = np.nan_to_num(A0_En, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sum(np.abs(A0_En) ** 2, axis=1)
    spectrum = 1.0 / (denom + 1e-30)

    # 3f. 全局寻峰
    peak_idx = np.argmax(spectrum)
    est_angle_deg = angles_deg[peak_idx]

    return est_angle_deg


# ============================================================
#  主程序：Monte Carlo 批量测试
# ============================================================
def main():
    N_TESTS = 50  # 测试次数

    print("=" * 60)
    print("  CSM DOA Monte Carlo Batch Test")
    print("=" * 60)
    print(f"  Number of tests: {N_TESTS}")
    print(f"  SNR: {SNR_DB} dB")
    print(f"  Array: ACA {SENSOR_ANGLES_DEG}")
    print(f"  Angle step: {ANGLE_STEP} deg")
    print()

    # 随机生成 N 个真实角度
    np.random.seed(42)
    true_angles = np.random.uniform(0, 360, N_TESTS)

    est_angles = np.zeros(N_TESTS)
    errors = np.zeros(N_TESTS)

    for i, true_angle in enumerate(true_angles):
        est_angle = evaluate_single_doa(true_angle)
        est_angles[i] = est_angle

        # 考虑 360° 环绕的误差计算
        raw_err = abs(est_angle - true_angle)
        err = min(raw_err, 360.0 - raw_err)
        errors[i] = err

        print(f"  [{i+1:2d}/{N_TESTS}] True: {true_angle:6.1f} deg -> "
              f"Est: {est_angle:6.1f} deg  (Error: {err:5.1f} deg)")

    # ---- 统计 ----
    mae = np.mean(errors)
    max_err = np.max(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    print()
    print("=" * 60)
    print("  Statistics")
    print("=" * 60)
    print(f"  Mean Absolute Error (MAE):     {mae:5.2f} deg")
    print(f"  Max Error:                     {max_err:5.2f} deg")
    print(f"  Root Mean Square Error (RMSE): {rmse:5.2f} deg")
    print(f"  Tests with error < 5 deg:      {np.sum(errors < 5.0):3d}/{N_TESTS}")
    print(f"  Tests with error < 10 deg:     {np.sum(errors < 10.0):3d}/{N_TESTS}")

    # ---- 散点图 ----
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(true_angles, est_angles, c='blue', s=30, alpha=0.7,
               edgecolors='k', linewidth=0.5, label='CSM Estimates')
    ax.plot([0, 360], [0, 360], 'r--', linewidth=1.5,
            label='Perfect Alignment (y=x)')

    ax.set_xlabel('True Angle (deg)', fontsize=12)
    ax.set_ylabel('Estimated Angle (deg)', fontsize=12)
    ax.set_title(f'CSM DOA: True vs Estimated Angle (N={N_TESTS}, SNR={SNR_DB} dB)',
                 fontsize=13)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_aspect('equal')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 在图上标注统计信息
    text_str = (f"MAE = {mae:.2f}°\n"
                f"Max Error = {max_err:.2f}°\n"
                f"RMSE = {rmse:.2f}°")
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIG_SCATTER, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Saved scatter plot: {FIG_SCATTER}")

    print("\n" + "=" * 60)
    print("  Batch Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
