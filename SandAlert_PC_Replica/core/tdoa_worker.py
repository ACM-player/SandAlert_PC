#!/usr/bin/env python3
"""
tdoa_worker.py
===============
TDOA 异步计算工作线程。

负责在独立 QThread 中安全运行 TDOAEngine 的 DOA 计算，
通过 PySide6 Signal 将结果回传给主 UI 线程，绝不阻塞 UI。

数据流：
  1. 主线程（SignalTracesTab）检测到有效撞击峰值
  2. 从多通道缓冲池截取 ±0.5ms 切片
  3. 调用 worker.enqueue(slice_4ch) 将任务投入队列
  4. 工作线程在后台完成 GCC-PHAT + 网格搜索
  5. 通过信号 doa_result 或 doa_error 回传结果

健壮性：
  - 内部耗时统计，超过代价门槛则发射 doa_likelihood_low
  - 绝不阻塞 UI 主线程

作者：AI Assistant
日期：2026-05-17
"""

import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from queue import Queue, Empty

from core.tdoa_engine import TDOAEngine


class TDOAWorker(QThread):
    """
    TDOA 异步计算工作线程。

    信号
    -------
    doa_result : Signal(float, float, object)
        发射参数：(est_angle_deg, cost_min, tdoa_obs)
        - est_angle_deg: 估计的方位角（度）
        - cost_min: 代价函数最小值（越小越可靠）
        - tdoa_obs: 观测时延向量 (numpy array, shape (m-1,))
    doa_likelihood_low : Signal(float)
        发射参数：(cost_min)
        当最小代价值超过阈值，表示似然极低、可能是杂波干扰。
    doa_error : Signal(str)
        发射参数：错误描述字符串
    """

    # ---- 似然门槛常量 ----
    COST_LIKELY_THRESHOLD = 1e-8   # 代价值低于此门槛才算可靠

    # ====== 信号定义 ======
    doa_result = Signal(float, float, object)
    doa_likelihood_low = Signal(float)
    doa_error = Signal(str)

    def __init__(
        self,
        fs: int = 2_000_000,
        radius: float = 0.05,
        c: float = 1500.0,
        parent=None,
    ):
        """
        初始化 TDOA 工作线程。

        参数
        ----------
        fs : int
            采样率（Hz），默认 2 MHz。
        radius : float
            阵列半径（m），默认 0.05 m。
        c : float
            声速（m/s），默认 1500 m/s。
        parent : QObject or None
            父对象。
        """
        super().__init__(parent)

        # ---- 持有一个独立的 TDOAEngine 实例 ----
        self._engine = TDOAEngine(
            fs=fs,
            radius=radius,
            c=c,
        )

        # ---- 线程安全的任务队列 ----
        self._task_queue: Queue = Queue()
        self._mutex = QMutex()
        self._is_running = False

    # ============================================================
    #  公开接口
    # ============================================================

    def enqueue(self, X: np.ndarray) -> None:
        """
        将一个 4 通道信号切片投入处理队列。
        该方法线程安全，可从主线程调用。

        参数
        ----------
        X : np.ndarray, shape (m, n_samples)
            4 通道信号切片，m 应为 4。
        """
        with QMutexLocker(self._mutex):
            self._task_queue.put(X.copy())

    def clear_queue(self) -> None:
        """清空所有待处理任务（例如切换模式时）。"""
        with QMutexLocker(self._mutex):
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except Empty:
                    break

    @property
    def pending_tasks(self) -> int:
        """当前队列中待处理的任务数。"""
        with QMutexLocker(self._mutex):
            return self._task_queue.qsize()

    def update_parameters(
        self,
        fs: int = None,
        radius: float = None,
        c: float = None,
    ) -> None:
        """
        在线更新物理参数（采样率、阵列半径、声速）。
        注意：参数更新后不会立即生效，下一次计算时使用新参数。

        参数
        ----------
        fs : int or None
            新的采样率。
        radius : float or None
            新的阵列半径。
        c : float or None
            新的声速。
        """
        if fs is not None:
            self._engine.fs = fs
        if radius is not None:
            self._engine.radius = radius
            # 重新计算传感器坐标
            self._engine.sensor_pos = np.array([
                [radius * np.cos(phi), radius * np.sin(phi)]
                for phi in np.deg2rad(self._engine.SENSOR_ANGLES_DEG)
            ])
        if c is not None:
            self._engine.c = c

    # ============================================================
    #  线程主循环
    # ============================================================

    def run(self):
        """
        线程主循环：持续轮询任务队列，处理 TDOA 计算。
        通过检查 _is_running 标志实现安全退出。
        """
        self._is_running = True

        while self._is_running:
            try:
                # 从队列中取任务，超时 200ms 以响应停止请求
                X = self._task_queue.get(timeout=0.2)
            except Empty:
                continue
            except Exception:
                continue

            # ---- 输入数据完整性检查 ----
            if X.ndim != 2 or X.shape[0] != 4:
                self.doa_error.emit(
                    f"输入数据维度错误: 期望 (4, n), 实际 {X.shape}"
                )
                continue

            if X.shape[1] < 10:
                self.doa_error.emit(
                    f"切片长度不足: {X.shape[1]} samples (需要至少 10)"
                )
                continue

            try:
                # ---- 执行一站式 DOA 估计 ----
                t0 = time.perf_counter()
                est_angle, tdoa_obs, angles, costs = (
                    self._engine.compute_doa(X)
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # ---- 评估估计可靠性 ----
                cost_min = float(np.min(costs))

                if cost_min > self.COST_LIKELY_THRESHOLD:
                    # 似然极低：可能是杂波偶然越过了阈值
                    # 不发射 doa_result，只发射低似然警告
                    self.doa_likelihood_low.emit(cost_min)
                else:
                    # 可靠估计 → 发射结果信号
                    self.doa_result.emit(est_angle, cost_min, tdoa_obs)

            except Exception as e:
                self.doa_error.emit(f"TDOA 计算异常: {str(e)}")

    # ============================================================
    #  安全停止
    # ============================================================

    def stop(self):
        """
        安全停止工作线程。
        - 设置停止标志位
        - 清空待处理队列
        - 等待线程退出（最多 2 秒）
        """
        self._is_running = False
        self.clear_queue()
        if self.isRunning():
            self.wait(2000)
