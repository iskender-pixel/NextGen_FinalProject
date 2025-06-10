import numpy as np
from scipy.fft import fft, fftshift
from scipy.io import loadmat
import matplotlib.pyplot as plt
from setup import *
from utils import *
import time
from scipy.signal import butter, filtfilt

#-----------INITIALIZE--------------INITIALIZE----------------------------
Pluto_IP = '192.168.2.1'
PlutoSamprate = 40e6
Tx_CenterFrequency = 1.5e9
Rx_CenterFrequency = 1.5e9
tx_gain = -10
rx_gain = 0

#-----------MAKE SIGNAL----------------MAKE SIGNAL-----------------------
B = 5e6
T = 100e-6
f0 = 1.5e9
fs = PlutoSamprate
Ts = 1/fs
t = np.arange(0, T-1/fs, 1/fs)
k = B / T;     # Chirp slope defined as ratio of bandwidth over duration
sig_A = np.exp(1j*2*np.pi*(0.5*k*t**2))
# sig_A_longer = np.array([])

sig_A_longer = np.exp(1j*B*t)

N = len(t)
rx_time_ms = 1000*T
plt.plot(sig_A)
plt.show()

my_sdr, tddn = initialize_Pluto_TDD(Pluto_IP, PlutoSamprate, f0, rx_gain, tx_gain, rx_time_ms)
save_path = os.path.dirname(os.path.abspath(__file__))
rx_SamplePerFrame = fs
print(my_sdr.rx)
tx_waveform = ((2**14))*sig_A_longer
frame_length_samples = np.floor(len(t))
# my_sdr = initialize_Pluto(Pluto_IP, np.int32(len(tx_waveform)), np.int32(PlutoSamprate), Tx_CenterFrequency, Rx_CenterFrequency, np.int32(rx_gain), np.int32(tx_gain),np.int32(rx_SamplePerFrame))
capture_range = 1
results = pluto_transmit_receive(my_sdr, tddn, tx_waveform, np.int32(capture_range), np.int32(frame_length_samples),save_path)
rx_data=loadmat(f'{save_path}/received_data.mat')

rx_data_sig=rx_data['received_data']
print(rx_data_sig)
rx_added = np.array([])
sig_a_new = np.array([])
for i in range(len(rx_data_sig)):
    rx_added = np.append(rx_added, rx_data_sig[i])
    # sig_a_new = np.append(sig_a_new, sig_A_longer)
    sig_a_new = sig_A_longer
# rx_added = rx_data_sig[0]
# plt.plot(rx_added)
# plt.show()

L= len(rx_added)
# print(L)
# print(Ts)
# t = np.arange(0, L-1/fs, 1/fs)
plt.plot(rx_added)
plt.show()
s_beat = rx_added * np.conj(sig_a_new)
nyq = 0.5 * fs 
normal_cutoff = B / nyq 
b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)

s_beat_filtered = filtfilt(b, a, np.real(s_beat))
sig_C=s_beat_filtered
plt.plot(sig_C)
plt.show()
f = np.linspace(-fs/2, fs/2, len(sig_C))
plt.plot(f,20*np.log10(np.abs(fftshift(fft(rx_added)))))
plt.show()
sig_C_sum=np.sum(sig_C,1)

    