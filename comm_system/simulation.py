import numpy as np

class UnderwaterChannel:
    """Realistic simulation of underwater communication effects"""
    def __init__(self,
                 sampling_rate=48000,         # Hz
                 multipath_profile='shallow', # 'shallow' or 'deep'
                 noise_power=1e-4,            # More realistic ambient noise
                 doppler_shift=5):            # Hz, low residual Doppler
        self.fs = sampling_rate
        self.noise_power = noise_power
        self.doppler_shift = doppler_shift
        self.multipath_delays = self._define_multipath(multipath_profile)

    def _define_multipath(self, profile):
        if profile == 'shallow':
            # Shallow water, strong early reflections
            return {0: 1.0, 5: 0.6, 12: 0.3, 30: 0.1}
        elif profile == 'deep':
            # Deep water, more spread out multipath
            return {0: 1.0, 20: 0.7, 50: 0.4, 100: 0.2}
        else:
            return {0: 1.0}

    def apply_effects(self, tx_signal):
        """Apply multipath, Doppler, and noise"""
        # Multipath convolution (simplified with roll)
        rx_signal = np.zeros_like(tx_signal)
        for delay, attenuation in self.multipath_delays.items():
            shifted = np.roll(tx_signal, delay)
            rx_signal += attenuation * shifted

        # Apply Doppler shift (small, linear)
        if self.doppler_shift:
            t = np.arange(len(rx_signal)) / self.fs
            rx_signal *= np.exp(1j * 2 * np.pi * self.doppler_shift * t)

        # Add ambient noise (AWGN)
        noise = np.sqrt(self.noise_power / 2) * (
            np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal))
        )

        return rx_signal + noise
