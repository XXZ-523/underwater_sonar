import numpy as np
from scipy.interpolate import interp1d

class AdaptiveEqualizer:
    """LMS Adaptive Equalizer for multipath compensation"""
    def __init__(self, filter_length=32, learning_rate=0.01):
        self.weights = np.zeros(filter_length, dtype=np.complex64)
        self.mu = learning_rate
        self.error_history = []

    def update(self, received, reference):
        error = reference - np.dot(self.weights, received)
        self.weights += self.mu * error * np.conj(received)
        self.error_history.append(abs(error))

    def equalize(self, signal):
        output = np.convolve(signal, self.weights[::-1], mode='same')
        return output

def ls_channel_estimation(rx_symbol, pilot_idx, pilots):
    return rx_symbol[pilot_idx] / pilots

def interpolate_channel(H_est, pilot_idx, N):
    interp_func = interp1d(pilot_idx, H_est, kind='linear', fill_value="extrapolate")
    return interp_func(np.arange(N))
