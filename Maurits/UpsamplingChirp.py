#_______LIBRARIES_________
import numpy as np
from scipy.fft import fft, fftshift
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample_poly, butter, filtfilt
import adi
from setupPlutoSDR_TDD import *
from TDD_Transreceiver import *
import time
import scipy.io
import os


#_____FUNCTIONS_________

def initialize_Pluto_TDD(PlutoIP, sample_rate, center_freq, rx_gain, tx_gain, frame_length_samples, bandwidth_offset=0):
    # %% Setup SDR
    PlutoIP = 'ip:' + PlutoIP
    my_sdr = adi.Pluto(uri=PlutoIP)
    tddn = adi.tddn(PlutoIP)

    # If you want repeatable alignment between transmit and receive then the rx_lo, tx_lo and sample_rate can only be set once after power up
    # But if you don't care about this, then program sample_rate and LO's as much as you want
    # So after bootup, check if the default sample_rate and LOs are being used, if so then program new ones!
    # if (
    #     30719990 < my_sdr.sample_rate < 30720009
    #     and 2399999990 < my_sdr.rx_lo < 2400000009
    #     and 2449999990 < my_sdr.tx_lo < 2450000009
    # ):
    my_sdr.sample_rate = int(sample_rate)
    my_sdr.rx_lo = int(center_freq + bandwidth_offset)
    my_sdr.tx_lo = int(center_freq + bandwidth_offset)
    print("Pluto has just booted and I've set the sample rate and LOs!")

    # Configure Rx
    my_sdr.rx_enabled_channels = [0]
    sample_rate = int(my_sdr.sample_rate)
    # manual or slow_attack
    my_sdr.gain_control_mode_chan0 = "manual"
    my_sdr.rx_hardwaregain_chan0 = int(rx_gain)
    # Default is 4 Rx buffers are stored, but to immediately see the result, set buffers=1
    my_sdr._rxadc.set_kernel_buffers_count(1)

    # Configure Tx
    my_sdr.tx_enabled_channels = [0]
    my_sdr.tx_hardwaregain_chan0 = int(tx_gain)
    my_sdr.tx_cyclic_buffer = True  # must be true to use the TDD transmit

    rx_time_ms = 1000 * frame_length_samples * (1 / my_sdr.sample_rate)
    N_rx = frame_length_samples
    my_sdr.rx_buffer_size = N_rx
    print(f"NRX = {N_rx}=============================")
    tddn.startup_delay_ms = 0
    print(my_sdr.sample_rate)
    print(frame_length_samples)
    tddn.frame_length_ms = ((1 / my_sdr.sample_rate) * frame_length_samples) * 1000
    tddn.burst_count = 0  # 0 means repeat indefinitely

    tddn.channel[0].on_raw = 0
    tddn.channel[0].off_raw = 0
    tddn.channel[0].polarity = 1
    tddn.channel[0].enable = 1

    # RX DMA SYNC
    tddn.channel[1].on_raw = 0
    tddn.channel[1].off_raw = 20
    tddn.channel[1].polarity = 0
    tddn.channel[1].enable = 1

    # TX DMA SYNC
    tddn.channel[2].on_raw = 0
    tddn.channel[2].off_raw = 20
    tddn.channel[2].polarity = 0
    tddn.channel[2].enable = 1

    tddn.sync_external = True  # enable external sync trigger
    tddn.enable = True  # enable TDD engine
    print("SDR Configuration Completed")

    # Return the my_sdr and tddn objects
    return my_sdr, tddn

def ReceiveSignal(fs, B, Bandwidth_offset=0):

    #initialization parameters.
    Pluto_IP = '192.168.2.1'

    Tx_CenterFrequency = 1.5e9
    Rx_CenterFrequency = 1.5e9
    tx_gain = -70
    rx_gain = 40
    save_path = os.path.dirname(os.path.abspath(__file__))

    #Signal parameters
    T = 400e-6
    f0 = 1.5e9
    Ts = 1 / fs
    t = np.arange(0, T - 1 / fs, 1 / fs)
    k = B / T  # Chirp slope defined as ratio of bandwidth over duration

    chirp_A = np.exp(1j * 2 * np.pi * (0.5 * k * t ** 2))

    samp_amount = len(t)

    tx_signal = ((2 ** 14)) * chirp_A

    my_sdr, tddn = initialize_Pluto_TDD(Pluto_IP, fs, f0, rx_gain, tx_gain, samp_amount, Bandwidth_offset)

    results = pluto_transmit_receive(my_sdr, tddn, tx_signal, 1, samp_amount, save_path)

    rx_data = loadmat(f'{save_path}/received_data.mat')
    rx_data_sig = rx_data['received_data']


    return rx_data_sig


def pluto_transmit_receive(my_sdr, tddn, iq, capture_range, frame_length_samples, path):
    # Invia il segnale IQ
    print("sending IQ data")

    iq = np.squeeze(iq)  # Removes dimensions of length 1

    my_sdr._rx_init_channels()
    my_sdr.tx(iq)

    # Enable software trigger for transmission
    tddn.sync_soft = 1  # Start the TDD transmit

    # Print configuration information
    print(f"TX/RX Sampling_rate: {my_sdr.sample_rate}")
    print(f"Number of samples in a frame: {frame_length_samples}")
    print(f"RX buffer length: {frame_length_samples}")
    print(f"TX buffer length: {len(iq)}")
    print(f"RX_receive time[ms]: {((1 / my_sdr.sample_rate) * frame_length_samples) * 1000}")
    print(f"TX_transmit time[ms]: {((1 / my_sdr.sample_rate) * len(iq)) * 1000}")
    print(f"TDD_frame time[ms]: {tddn.frame_length_ms}")
    print(f"TDD_frame time[raw]: {tddn.frame_length_raw}")

    # Initialize the array used to store the received data
    received_array = np.zeros((capture_range, my_sdr.rx_buffer_size), dtype=complex) * 1j

    # Receive data
    for r in range(20):
        received_array = my_sdr.rx()

    # Shutdown Pluto transmission
    tddn.enable = 0
    for i in range(3):
        tddn.channel[i].on_ms = 0
        tddn.channel[i].off_raw = 0
        tddn.channel[i].polarity = 0
        tddn.channel[i].enable = 1
    tddn.enable = 1
    tddn.enable = 0

    # Clear the transmission buffer
    my_sdr.tx_destroy_buffer()
    print("Pluto Buffer Cleared!")

    # Save
    mat_file_path = os.path.join(path, 'received_data.mat')
    scipy.io.savemat(mat_file_path, {'received_data': received_array})

    print(f"Data saved to {mat_file_path}")

    # Return the received_array list
    return received_array.tolist()

def plot(fs, signal):
    f, t_spec, Sxx = spectrogram(np.real(signal), fs=fs, nperseg=1024, noverlap=512)

    # Convert power to decibels
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)

    # Plot 2D spectrogram
    plt.figure(figsize=(12, 6))

    plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='viridis')
    plt.title('2D Spectrogram of rx_added')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power [dB]')
    plt.tight_layout()
    plt.show()

    return 0


def main():
    PlutoSamprate = 40e6
    B = 5e6

    fAxis = PlutoSamprate

    plot(fAxis, ReceiveSignal(PlutoSamprate, B, 0))


    return 0


if __name__ == "__main__":

    #Initialized the pluto SDR

    main()