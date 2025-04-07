import numpy as np
from scipy.fft import ifft

class SonarModem:
    """OFDM-based Sonar Modem with Pilot Insertion and Adaptive Modulation"""

    def __init__(self, carrier_freq=25e3, bandwidth=10e3, subcarriers=64, cyclic_prefix=16):
        self.carrier_freq = carrier_freq
        self.bandwidth = bandwidth
        self.subcarriers = subcarriers
        self.cyclic_prefix = cyclic_prefix
        self.last_ofdm_symbol = None

    def qam_constellation(self, M):
        if M == 2:  # BPSK
            return np.array([1, -1])
        elif M == 4:  # QPSK
            return np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        elif M == 16:  # 16-QAM
            real = np.array([-3, -1, 1, 3])
            imag = np.array([-3, -1, 1, 3])
            grid = [x + 1j*y for x in real for y in imag]
            return np.array(grid) / np.sqrt(10)
        else:
            raise ValueError("Unsupported modulation order")

    def bits_to_symbols(self, bits, M):
        k = int(np.log2(M))
        bits = bits[:len(bits) - len(bits) % k]
        symbol_indices = bits.reshape((-1, k)).dot(1 << np.arange(k)[::-1])
        return self.qam_constellation(M)[symbol_indices]

    def symbols_to_bits(self, symbols, M):
        const = self.qam_constellation(M)
        dists = abs(symbols.reshape(-1, 1) - const)**2
        indices = np.argmin(dists, axis=1)
        k = int(np.log2(M))
        bits = ((indices[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)
        return bits.flatten()

    def insert_pilots(self, symbols, pilot_interval=4):
        N = self.subcarriers
        ofdm_symbol = np.zeros(N, dtype=complex)
        pilots = np.ones(N // pilot_interval, dtype=complex)
        data_idx = [i for i in range(N) if i % pilot_interval != 0]
        pilot_idx = [i for i in range(N) if i % pilot_interval == 0]

        if len(symbols) != len(data_idx):
            raise ValueError(f"Expected {len(data_idx)} symbols, got {len(symbols)}")

        ofdm_symbol[pilot_idx] = pilots
        ofdm_symbol[data_idx] = symbols
        return ofdm_symbol, pilot_idx, data_idx, pilots

    def transmit_with_pilots(self, bits, snr_dB=15, pilot_interval=4):
        M = self.select_modulation(snr_dB)
        k = int(np.log2(M))
        num_data_subcarriers = self.subcarriers - self.subcarriers // pilot_interval
        max_bits = num_data_subcarriers * k
        bits = bits[:max_bits]  # Trim bits to fit symbol count

        symbols = self.bits_to_symbols(bits, M)
        ofdm_symbol, pilot_idx, data_idx, pilots = self.insert_pilots(symbols, pilot_interval)
        self.last_ofdm_symbol = ofdm_symbol
        tx_time = ifft(ofdm_symbol)
        return tx_time, pilot_idx, data_idx, pilots

    def demodulate(self, symbols, snr_dB=15):
        M = self.select_modulation(snr_dB)
        return self.symbols_to_bits(symbols, M)

    def select_modulation(self, snr_dB):
        if snr_dB > 20:
            return 16
        elif snr_dB > 10:
            return 4
        else:
            return 2

