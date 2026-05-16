import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class DSPEngine:
    """
    核心数字信号处理引擎 (DSP)
    负责数据滤波、包络提取、寻峰及出砂量计算
    """
    def __init__(self, sample_rate=1000):
        # 假设采样率为 1000Hz (每秒 1000 个数据点)
        self.fs = sample_rate
        self.nyq = 0.5 * self.fs
        
        # 设计一个 4 阶巴特沃斯高通滤波器
        # 截止频率设为 20Hz，过滤掉缓慢的直流漂移（如 -2.85V 的基线）和低频震动
        cutoff_freq = 20.0 
        normal_cutoff = cutoff_freq / self.nyq
        self.b, self.a = butter(4, normal_cutoff, btype='high', analog=False)
        
        # ====== 防翻车机制：定义异常波峰阈值 ======
        # 【重要说明】防插拔脉冲的逻辑不能放在滤波之前！
        # 因为现场原始信号存在 -2.85V 左右的直流偏置（DC Offset），
        # 如果在入口处做 np.clip，会把正常波动"剃平"成死线，
        # 高通滤波器处理直线输出恒为 0，导致系统"脑死亡"。
        #
        # 正确的做法：在滤波、整流完成、基线归零之后，
        # 在寻峰阶段过滤掉振幅 > 2.0V 的异常波峰。
        # 此时基线在 0V 附近，>2.0V 的波峰一定是传感器插拔等异常产生的。
        self.PEAK_AMPLITUDE_LIMIT = 2.0  # 整流后波峰振幅超过 2.0V 视为异常，直接丢弃

    def process_window(self, raw_voltage_array, threshold_margin, k_v2):
        """
        处理一个窗口的数据
        返回: (滤波后的信号, 动态阈值, IPS, 出砂量)
        """
        if len(raw_voltage_array) < 15: # 数据太少无法滤波
            return raw_voltage_array, 0, 0, 0.0

        # 1. 零相位高通滤波 (消除 DC 偏置，基线将完美归零)
        # 【注意】原始信号有 -2.85V 直流偏置，绝对不能提前做 clip！
        # 直接送入 filtfilt，高通滤波器会自动去除直流分量，基线归零。
        filtered_signal = filtfilt(self.b, self.a, raw_voltage_array)
        
        # 2. 信号整流 (取绝对值，将向下的负向突刺也翻转上来，方便寻峰)
        rectified_signal = np.abs(filtered_signal)
        
        # 3. 计算本底与动态阈值 (此时基线在 0 附近)
        sb = np.mean(rectified_signal)
        dynamic_threshold = sb + threshold_margin
        
        # 4. 寻峰 (只找突刺)
        peaks, peak_properties = find_peaks(rectified_signal, height=dynamic_threshold, distance=10)
        
        # ====== 防翻车机制：在寻峰之后过滤异常波峰 ======
        # 此时基线已经归零（在 0V 附近），整流后的波峰振幅如果 > 2.0V，
        # 说明是传感器插拔等硬件异常产生的极端脉冲，不是真实的砂粒撞击信号。
        # 使用列表推导式剔除这些异常波峰，只保留有效波峰参与能量计算。
        if len(peaks) > 0:
            valid_peaks = [p for p in peaks if rectified_signal[p] < self.PEAK_AMPLITUDE_LIMIT]
        else:
            valid_peaks = []
        
        # 5. 解耦计算出砂量与能量
        ips = len(valid_peaks)
        mass_rate = 0.0
        window_energy = 0.0  # 记录当前窗口的纯声学能量
        
        if ips > 0:
            sp = rectified_signal[valid_peaks]
            # 计算纯粹的声学能量 SUM(sp - sb)^2
            window_energy = np.sum((sp - sb)**2)
            # 瞬时出砂量 M = 能量 / (K*v^2)
            mass_rate = window_energy / k_v2
            
        # 注意这里返回了 5 个值，多加了一个 window_energy
        return rectified_signal, dynamic_threshold, ips, mass_rate, window_energy
