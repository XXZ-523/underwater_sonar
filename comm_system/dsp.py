import numpy as np

class AdaptiveEqualizer:
    """LMS Adaptive Equalizer for multipath compensation"""
    def __init__(self, filter_length=32, learning_rate=0.01):
        self.weights = np.zeros(filter_length, dtype=np.complex64)
        self.mu = learning_rate
        self.error_history = []

    def update(self, received, reference):
        """Update filter weights"""
        error = reference - np.dot(self.weights, received)
        self.weights += self.mu * error * np.conj(received)
        self.error_history.append(np.abs(error) ** 2)
        return np.abs(error) ** 2

    def equalize(self, signal):
        """Apply learned weights to signal"""
        return np.convolve(signal, self.weights[::-1], mode='same')