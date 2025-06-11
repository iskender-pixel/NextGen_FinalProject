import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.fft import fft, fftshift
from scipy import signal
from scipy.signal import spectrogram
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# Generating TX chirp
B = 30e6
T = 100e-6
f0 = 1.5e9
fs = 61e6
Ts = 1/fs
t = np.arange(0, T-1/fs, 1/fs)
k = B / T      # Chirp slope defined as ratio of bandwidth over duration
chirp_A = np.exp(1j*2*np.pi*(0.5*k*t**2))
number_of_measurements = 1
sig_A = chirp_A



# plt.plot(t, np.real(sig_A))
#
#
# plt.show()
# f = np.linspace(0, fs, len(sig_A))
# plt.plot(f, abs(fft(sig_A)))
# plt.show()
#
#
#
# f, t_spec, Sxx = spectrogram(np.real(sig_A), fs=fs, nperseg=1024, noverlap=512)
# Sxx_dB = 10*np.log10(Sxx)
# plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='viridis')
# plt.show()

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

path = "Close_B30e6/received_data.mat"

def plotSbeat(path):
    rx_data = loadmat(f'{path}')['received_data'].flatten()[0:len(t)*99]
    rx = loadmat(f'{path}')['received_data']
    print(rx.shape)
    rx_data_matrix = np.array_split(rx_data, 99)
    s_beat = np.zeros(len(t), dtype=complex)
    for i in range(99):

        s_beat += rx_data_matrix[i] * np.conj(sig_A)


    rx_data_added = np.zeros((len(rx[0,:])), dtype=complex)

    print(len(rx[:,0]))
    print(len(rx[0,:]))
    for i in range(len(rx[:,0])):
        rx_data_added += rx[i,:]

    print(rx_data_added)
    # s_beat = rx_data_matrix[0] * np.conj(sig_A)

    # plt.plot(t, np.real(s_beat))
    # plt.show()

    # tnew = np.arange(0, 604900/fs, 1/fs)
    # plt.plot(t, np.real(rx_data_matrix[0]))
    # plt.show()

    nyq = 0.5 * fs
    normal_cutoff = B / nyq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)

    s_beat_filtered = filtfilt(b, a, np.real(s_beat))


    N_fft = 2**nextpow2(len(rx_data))*2

    freq_tone = np.abs(fftshift(fft(s_beat_filtered,N_fft)))
    freq_axis = np.linspace(-fs/2, fs/2, len(freq_tone))
    range_axis= (3e8 * freq_axis) / (2 * k)
    plt.plot(range_axis, 10*np.log10(freq_tone), label=f"{path}")
    plt.xlabel("Range axis (m)")
    plt.xlim(0, 300)



if __name__ == '__main__':

    plotSbeat("Close_B30e6/received_data.mat")
    plotSbeat("3m_B30e6/received_data.mat")
    plotSbeat("6m_B30e6/received_data.mat")
    plotSbeat("No_reflection_B30e6/received_data.mat")
    plt.legend()
    plt.show()


