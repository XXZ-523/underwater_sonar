import numpy as np

class UnderwaterChannel:
    """Simulates underwater communication effects"""
    def __init__(self, multipath_delays=None, noise_power=0.0001, doppler_shift=0):
        self.multipath_delays = multipath_delays or {0: 1.0}  # No multipath
        self.noise_power = noise_power  # Reduced noise
        self.doppler_shift = doppler_shift  # No Doppler shift

    def apply_effects(self, tx_signal):
        """Add multipath, noise, and Doppler effects"""
        # Multipath
        rx_signal = tx_signal.copy()
        for delay, attenuation in self.multipath_delays.items():
            rx_signal += attenuation * np.roll(tx_signal, delay)

        # Doppler (disabled for now)
        t = np.arange(len(rx_signal)) / (2 * self.doppler_shift) if self.doppler_shift else 0
        if self.doppler_shift:
            rx_signal *= np.exp(1j * 2 * np.pi * self.doppler_shift * t)

        # Add noise
        noise = np.sqrt(self.noise_power / 2) * (
            np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
        return rx_signal + noise