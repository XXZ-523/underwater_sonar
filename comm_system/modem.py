import numpy as np
from scipy.fft import fft, ifft

class SonarModem:
    """OFDM-based Sonar Modem with DSP enhancements"""
    def __init__(self, carrier_freq=25e3, bandwidth=10e3, subcarriers=64, cyclic_prefix=16):
        self.carrier_freq = carrier_freq
        self.bandwidth = bandwidth
        self.subcarriers = subcarriers
        self.cyclic_prefix = cyclic_prefix

        # QPSK constellation
        self.constellation = {
            0: (1 + 1j) / np.sqrt(2),
            1: (-1 + 1j) / np.sqrt(2),
            2: (-1 - 1j) / np.sqrt(2),
            3: (1 - 1j) / np.sqrt(2)
        }

    def transmit(self, data_bits, return_symbols=False):
        """Convert bits to OFDM sonar signal"""
        if len(data_bits) % 2 != 0:
            raise ValueError("Number of bits must be even for QPSK modulation")

        # QPSK modulation
        symbols = np.array([self.constellation[int(f"{b1}{b2}", 2)]
                            for b1, b2 in zip(data_bits[::2], data_bits[1::2])])

        if return_symbols:
            return symbols  # Return symbols for visualization

        # OFDM modulation
        ofdm_symbols = ifft(symbols, self.subcarriers)
        tx_signal = np.concatenate([ofdm_symbols[-self.cyclic_prefix:], ofdm_symbols])
        return np.real(tx_signal)

    def receive(self, rx_signal, return_symbols=False):
        """Demodulate received sonar signal"""
        # Remove cyclic prefix
        cp_removed = rx_signal[self.cyclic_prefix:]

        # OFDM demodulation
        symbols = fft(cp_removed, self.subcarriers)

        if return_symbols:
            return symbols  # Return symbols for visualization

        # QPSK demodulation
        rx_bits = []

        for symbol in symbols:
            # Find the closest constellation point
            distances = [np.abs(symbol - point) for point in self.constellation.values()]
            closest = np.argmin(distances)
            # Convert to bits (2 bits per symbol)
            rx_bits.extend([int(bit) for bit in f"{closest:02b}"])
        return np.array(rx_bits)
        rx_bits = modem.receive(cleaned_signal)
        print(f"Number of received bits: {len(rx_bits)}")