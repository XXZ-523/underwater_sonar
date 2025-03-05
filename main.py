import numpy as np
import matplotlib.pyplot as plt
from comm_system.modem import SonarModem
from comm_system.simulation import UnderwaterChannel
from comm_system.dsp import AdaptiveEqualizer

def main():
    # Initialize components
    modem = SonarModem()
    channel = UnderwaterChannel()  # Simplified channel
    equalizer = AdaptiveEqualizer()

    # Generate test data
    num_bits = 256  # Must be even
    data_bits = np.random.randint(0, 2, num_bits)

    # Transmit
    tx_signal = modem.transmit(data_bits)

    # Simulate channel
    rx_signal = channel.apply_effects(tx_signal)

    # Equalization
    training_length = 32
    known_preamble = tx_signal[:training_length]
    for i in range(training_length):
        equalizer.update(rx_signal[i:i + len(equalizer.weights)], known_preamble[i])
    cleaned_signal = equalizer.equalize(rx_signal)

    # Receive
    received_bits = modem.receive(cleaned_signal)

    # Performance analysis
    if len(received_bits) != len(data_bits):
        print(f"Warning: Received {len(received_bits)} bits, expected {len(data_bits)}")
        # Truncate or pad to match lengths
        min_length = min(len(received_bits), len(data_bits))
        received_bits = received_bits[:min_length]
        data_bits = data_bits[:min_length]

    ber = np.mean(received_bits != data_bits)
    print(f"Bit Error Rate: {ber:.4f}")

    # Debugging: Plot constellations
    tx_symbols = modem.transmit(data_bits, return_symbols=True)
    rx_symbols = modem.receive(rx_signal, return_symbols=True)
    plt.scatter(np.real(tx_symbols), np.imag(tx_symbols), label="Transmitted")
    plt.scatter(np.real(rx_symbols), np.imag(rx_symbols), label="Received")
    plt.legend()
    plt.title("Transmitted vs Received Constellation")
    plt.show()
if __name__ == "__main__":
    main()