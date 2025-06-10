import numpy as np
from scipy.fft import fft, fftshift
from scipy.io import loadmat
import matplotlib.pyplot as plt

from setupPlutoSDR_TDD import *
from TDD_Transreceiver import *
import time
from scipy.signal import butter, filtfilt

#-----------INITIALIZE--------------INITIALIZE----------------------------
Pluto_IP = '192.168.2.1'
PlutoSamprate = 61e6
Tx_CenterFrequency = 1.5e9
Rx_CenterFrequency = 1.5e9
tx_gain = -80
rx_gain = 40
save_path =  os.path.dirname(os.path.abspath(__file__))
#-----------MAKE SIGNAL----------------MAKE SIGNAL-----------------------
B = 30e6
T = 400e-6
f0 = 1.5e9
fs = PlutoSamprate
Ts = 1/fs
t = np.arange(0, T-1/fs, 1/fs)
k = B / T     # Chirp slope defined as ratio of bandwidth over duration
alpha = 1e6
print(alpha)
chirp = np.exp(1j*2*np.pi*(0.5*k*t**2))
# chirp = np.exp(1j*2*np.pi*(0.5*k*t**2))*t
number_of_measurements = 1
sig_A = chirp
sig_tx = np.array([])
for i in range(number_of_measurements):
    sig_tx = np.append(sig_tx, chirp)

samp_amount = (number_of_measurements*len(t))
tx_signal = ((2**13))*sig_tx

my_sdr, tddn = initialize_Pluto_TDD(Pluto_IP, fs, f0, rx_gain, tx_gain, samp_amount)
results = pluto_transmit_receive(my_sdr, tddn, tx_signal, 1, samp_amount,save_path)

rx_data=loadmat(f'{save_path}/received_data.mat')

rx_data_sig=rx_data['received_data']
rx_added = np.array(rx_data_sig).squeeze()
# noise = np.random.normal(loc=0.0, scale=20, size=rx_added.shape)

# rx_added = rx_added + noise
s_beat = rx_added * np.conj(sig_tx)
nyq = 0.5 * fs 
normal_cutoff = B / nyq 
b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
s_beat_filtered = filtfilt(b, a, np.real(s_beat))

freq_axis = np.linspace(-fs/2, fs/2, len(s_beat_filtered))

freq_tone = np.abs(fftshift(fft(s_beat_filtered)))

time_axis_new = np.tile(t, number_of_measurements)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0,0].plot(time_axis_new, np.real(sig_tx))
axs[0,0].set_title('Input signal')

axs[0,1].plot(time_axis_new, np.real(rx_added))
axs[0,1].set_title("received signal")

axs[1,0].plot(time_axis_new, s_beat_filtered)
axs[1,0].set_title('Mixed signal (time)')

axs[1,1].plot(freq_axis, freq_tone)
axs[1,1].set_title("received signal")

print(f'The maximum frequency is at :{freq_axis[np.argmax(freq_tone)]} Hz')
plt.show()

from scipy.signal import spectrogram

# Compute spectrogram
f, t_spec, Sxx = spectrogram(np.real(rx_added), fs=fs, nperseg=1024, noverlap=512)

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
