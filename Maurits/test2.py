from scipy.signal import freqz
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
B = 5e6
T = 400e-6
f0 = 1.5e9
fs = 40e6
# Ts = 1/fs
low_cutoff = 20         # Hz
high_cutoff = B         # Hz (your signal bandwidth)
nyq = 0.5 * fs          # Nyquist frequency
low = low_cutoff / nyq
high = high_cutoff / nyq

# Sanity check: ensure high < 1.0
if high >= 1.0:
    raise ValueError("High cutoff exceeds Nyquist. Adjust B or fs.")

# Design Butterworth bandpass filter
b, a = butter(N=4, Wn=[low, high], btype='band')

w, h = freqz(b, a, worN=8000)
plt.plot((fs * 0.5 / np.pi) * w, abs(h))
plt.title('Bandpass Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid()
plt.show()
