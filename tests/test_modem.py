import numpy as np
from comm_system.modem import SonarModem

def test_modem_transmit_receive():
    modem = SonarModem()
    data_bits = np.random.randint(0, 2, 256)
    tx_signal = modem.transmit(data_bits)
    rx_signal = tx_signal  # No channel effects
    received_bits = modem.receive(rx_signal)
    assert np.array_equal(received_bits, data_bits), "Transmit/Receive failed"