#!/usr/bin/env python3
"""
batch_test_tdoa.py
==================
SandAlert TDOA V2 — Monte Carlo 批量鲁棒性验证

对 N=50 个随机 0~360° 方位角进行全空间扫描测试，
验证 GCC-PHAT + Grid Search 架构的全局定位精度。
"""

import numpy as np
from scipy.signal import chirp
from scipy.fft import fft, ifft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# ============================================================
#  全局参数配置（与 tdoa_estimator.py 完全一致）
# ============================================================
FS = 2_000_000
DURATION = 0.005
N_SAMPLES = int(FS * DURATION)

M = 4
RADIUS = 0.05
SOUND_SPEED = 1500.0

F0 = 100e3
F1 = 300e3

ANGLE_STEP = 0.5
REF_CHANNEL = 0

N_TRIALS = 50
SNR_DB = 10.0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_SCATTER = os.path.join(_SCRIPT_DIR, "tdoa_scatter.png")


# ============================================================
#  传感器布局（UCA）
# ============================================================
SENSOR_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])
SENSOR_ANGLES_RAD = np.deg2rad(SENSOR_ANGLES_DEG)
SENSOR_POS = np.array([
    [RADIUS * np.cos(phi), RADIUS * np.sin(phi)]
    for phi in SENSOR_ANGLES_RAD
])


# ============================================================
#  GCC-PHAT 时延估计
# ============================================================
def gcc_phat(sig1, sig2, fs):
    """GCC-PHAT 时延估计。返回正数表示 sig2 比 sig1 晚到。"""
    n = len(sig1)
    X1 = fft(sig1)
    X2 = fft(sig2)
    G12 = np.conj(X1) * X2
    G12_phat = G12 / (np.abs(G12) + 1e-30)
    cross_corr = np.real(ifft(G12_phat))
    cross_corr = np.fft.fftshift(cross_corr)
    peak_idx = np.argmax(np.abs(cross_corr))
    delay_samples = peak_idx - n // 2
    delay_sec = delay_samples / fs
    return delay_sec


def compute_tdoa_vector(X, fs):
    """以参考通道为基准，计算所有通道的 TDOA 向量。"""
    m = X.shape[0]
    ref_signal = X[REF_CHANNEL]
    tdoa_obs = np.zeros(m - 1)
    for i in range(m):
        if i == REF_CHANNEL:
            continue
        delay_sec = gcc_phat(ref_signal, X[i], fs)
        ch_idx = i - 1 if i > REF_CHANNEL else i
        tdoa_obs[ch_idx] = delay_sec
    return tdoa_obs


# ============================================================
#  理论时延计算
# ============================================================
def compute_theoretical_tdoa(theta_deg):
    """计算候选角度 theta 下的理论相对时延向量。"""
    theta_rad = np.deg2rad(theta_deg)
    source_dir = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    delays = -(SENSOR_POS @ source_dir) / SOUND_SPEED
    tau_theory = np.delete(delays, REF_CHANNEL) - delays[REF_CHANNEL]
    return tau_theory


# ============================================================
#  网格搜索 DOA 估计
# ============================================================
def grid_search_doa(tdoa_obs):
    """网格搜索方位角估计。"""
    angles_deg = np.arange(0, 360, ANGLE_STEP)
    cost_values = np.zeros(len(angles_deg))
    for i, theta in enumerate(angles_deg):
        tau_theory = compute_theoretical_tdoa(theta)
        cost_values[i] = np.sum((tdoa_obs - tau_theory) ** 2)
    return angles_deg[np.argmin(cost_values)]


# ============================================================
#  单次 DOA 估计评估
# ============================================================
def evaluate_single_doa(true_angle_deg, snr_db=SNR_DB):
    """对单个真实角度执行完整的 DOA 估计流程。"""
    # ---- 1. 生成阵列信号 ----
    theta_true = np.deg2rad(true_angle_deg)
    source_dir = np.array([np.cos(theta_true), np.sin(theta_true)])
    delays_true = -(SENSOR_POS @ source_dir) / SOUND_SPEED

    t = np.arange(N_SAMPLES) / FS
    t0 = 0.001
    sigma = 0.0003
    envelope = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    source_signal = envelope * chirp(t, f0=F0, t1=t0 + 2 * sigma,
                                     f1=F1, method='linear')
    source_signal[t < t0 - 3 * sigma] = 0.0
    source_signal[t > t0 + 3 * sigma] = 0.0

    X = np.zeros((M, N_SAMPLES))
    for i in range(M):
        delay_samples = int(round(delays_true[i] * FS))
        if delay_samples >= 0:
            X[i, delay_samples:] = source_signal[:N_SAMPLES - delay_samples]
        else:
            X[i, :delay_samples] = source_signal[-delay_samples:]

    signal_power = np.mean(source_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(M, N_SAMPLES)
    X_noisy = X + noise

    # ---- 2. GCC-PHAT 时延估计 ----
    tdoa_obs = compute_tdoa_vector(X_noisy, FS)

    # ---- 3. 网格搜索 DOA ----
    est_angle_deg = grid_search_doa(tdoa_obs)

    # ---- 计算误差（处理 360 环绕） ----
    raw_error = abs(est_angle_deg - true_angle_deg)
    if raw_error > 180:
        raw_error = 360 - raw_error
    error_deg = raw_error

    # ---- 真值相对时延 ----
    delays_true_rel = np.delete(delays_true, REF_CHANNEL) - delays_true[REF_CHANNEL]

    return est_angle_deg, error_deg, tdoa_obs, delays_true_rel


# ============================================================
#  Monte Carlo 批量测试
# ============================================================
def run_monte_carlo():
    """执行 N 次随机角度 Monte Carlo 测试。"""
    print("=" * 65)
    print("  SandAlert TDOA V2 - Monte Carlo Robustness Verification")
    print("=" * 65)
    print(f"  FS = {FS/1e6:.1f} MHz  |  F0 = {F0/1e3:.0f} kHz  |  F1 = {F1/1e3:.0f} kHz")
    print(f"  Array: UCA (0, 90, 180, 270) deg  |  SNR = {SNR_DB:.0f} dB")
    print(f"  Trials: N = {N_TRIALS}  |  Grid step = {ANGLE_STEP} deg")
    print("=" * 65)

    np.random.seed(42)
    true_angles = np.random.uniform(0, 360, N_TRIALS)
    est_angles = np.zeros(N_TRIALS)
    errors = np.zeros(N_TRIALS)

    print(f"\n  Running Monte Carlo simulation...\n")
    start_time = time.time()

    for trial_idx in range(N_TRIALS):
        true_angle = true_angles[trial_idx]
        est_angle, error_deg, _, _ = evaluate_single_doa(true_angle, SNR_DB)
        est_angles[trial_idx] = est_angle
        errors[trial_idx] = error_deg
        print(f"    [{trial_idx+1:2d}/{N_TRIALS}] "
              f"True: {true_angle:7.1f} deg  ->  "
              f"Est: {est_angle:7.1f} deg  "
              f"(Error: {error_deg:5.1f} deg)")

    elapsed = time.time() - start_time

    # ---- 统计指标 ----
    mae = np.mean(errors)
    max_err = np.max(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    success_rate = np.mean(errors <= ANGLE_STEP) * 100

    print(f"\n  {'='*55}")
    print(f"  Monte Carlo Results Summary")
    print(f"  {'='*55}")
    print(f"  Total trials:         {N_TRIALS}")
    print(f"  Elapsed time:         {elapsed:.1f} s ({elapsed/N_TRIALS:.2f} s/trial)")
    print(f"  {'-'*55}")
    print(f"  Mean Absolute Error (MAE):      {mae:.4f} deg")
    print(f"  Root Mean Square Error (RMSE):  {rmse:.4f} deg")
    print(f"  Max Error:                      {max_err:.4f} deg")
    print(f"  {'-'*55}")
    print(f"  Success rate (error <= {ANGLE_STEP} deg):  {success_rate:.1f} %")
    print(f"  {'='*55}")

    # 打印异常点
    large_errors = errors > 1.0
    if np.any(large_errors):
        print(f"\n  Outliers (error > 1.0 deg):")
        for idx in np.where(large_errors)[0]:
            print(f"    True: {true_angles[idx]:7.1f} deg  ->  "
                  f"Est: {est_angles[idx]:7.1f} deg  "
                  f"(Error: {errors[idx]:.2f} deg)")

    return true_angles, est_angles, errors


# ============================================================
#  可视化
# ============================================================
def plot_scatter(true_angles, est_angles, errors):
    """真实角度 vs 估计角度散点图。"""
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(true_angles, est_angles, c=errors,
                         cmap='viridis', s=50, alpha=0.8,
                         edgecolors='black', linewidth=0.5,
                         vmin=0, vmax=min(np.max(errors) + 0.1, 2.0))

    ax.plot([0, 360], [0, 360], 'r--', linewidth=1.5, alpha=0.7,
            label='y = x (Perfect)')

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel('True Azimuth (deg)', fontsize=12)
    ax.set_ylabel('Estimated Azimuth (deg)', fontsize=12)
    ax.set_title(
        f'TDOA-GCC-PHAT Monte Carlo Validation (N={N_TRIALS}, SNR={SNR_DB:.0f} dB)\n'
        f'MAE = {np.mean(errors):.3f} deg  |  Max Error = {np.max(errors):.3f} deg',
        fontsize=13)

    cbar = plt.colorbar(scatter, ax=ax, label='Absolute Error (deg)')
    cbar.ax.tick_params(labelsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(FIG_SCATTER, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Saved scatter plot: {FIG_SCATTER}")


# ============================================================
#  主程序
# ============================================================
def main():
    true_angles, est_angles, errors = run_monte_carlo()
    plot_scatter(true_angles, est_angles, errors)
    print(f"\n  Monte Carlo validation complete!\n")


if __name__ == "__main__":
    main()
