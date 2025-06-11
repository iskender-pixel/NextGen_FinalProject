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
T = 800e-6
f0 = 1.5e9
fs = 61e6
Ts = 1/fs
t = np.arange(0, T-1/fs, 1/fs)
Tr = 2*T
t_triangle = np.arange(0, Tr-1/fs, 1/fs)

k = B / T      # Chirp slope defined as ratio of bandwidth over duration
Chirp1 = np.exp(1j*2*np.pi*(0.5*k*t**2))
Chirp2 = np.exp(1j*2*np.pi*(0.5*k*np.flip(t)**2))

Triangle = np.append(Chirp1[:-1], Chirp2)

f, t_spec, Sxx = spectrogram(np.real(Triangle), fs=fs, nperseg=1024, noverlap=512)

# Convert power to decibels
Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)

# Plot 2D spectrogram
plt.figure(figsize=(12, 6))

plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='viridis')
plt.title('2D Spectrogram of triangle chirp')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()

plt.plot(t_triangle, np.real(Triangle))
plt.title("time domain triangle chirp")
plt.xlabel('Time [s]')
plt.show()

B = 60e6
T = 800e-6
f0 = 1.5e9
fs = 61e6
Ts = 1/fs
t = np.arange(0, T-1/fs, 1/fs)


k = B / T      # Chirp slope defined as ratio of bandwidth over duration
Chirp1 = np.exp(1j*2*np.pi*(0.5*k*t**2))



f, t_spec, Sxx = spectrogram(np.real(Chirp1), fs=fs, nperseg=1024, noverlap=512)

# Convert power to decibels
Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)

# Plot 2D spectrogram
plt.figure(figsize=(12, 6))

plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='viridis')
plt.title('2D Spectrogram of triangle chirp')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()

plt.plot(t, np.real(Chirp1))
plt.title("time domain triangle chirp")
plt.xlabel('Time [s]')
plt.show()