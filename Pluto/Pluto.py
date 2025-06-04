from GivenFiles.setupPlutoSDR_TDD import initialize_Pluto_TDD
from GivenFiles.TDD_Transreceiver import pluto_transmit_receive
import numpy as np
import matplotlib.pyplot as plt
import math as m

Path = "Pluto/Received Data"
Pluto_IP = '192.168.2.1'
PlutoSamprate = 60.5e6  # Sampling frequency (Hz): Pluto can sample up to 61 MHz but due to the USB 2.0 interface you have to choose values lower or equal to 5MHz if you want toreceive 100# of samples over time.
centerFrequency = 2.5e9  #Pluto operating frequency (Hz) must be between 70MHz and 6GHz
#
tx_gain = -20  #Pluto TX channel Gain must be between 0 and -88
rx_gain = 40   #Pluto RX channel Gain must be between -3 and 70


# ----CHIRP GENERATION-----
B = 30e6              # Chirp bandwidth (Hz)
T = 100e-6            # Chirp duration (s)
f0 = 2.5e9            # Carrier frequency of the RF signal (Hz)
fs = PlutoSamprate    # Sampling rate of all the signals in this simulation (Hz)
c = 3e8               # Speed of light (m/s)

chirp_duration = 1  # Chirp duration (ms)
t = np.arange(0, chirp_duration*1e-3,1/fs)

k = B / T    # Chirp slope defined as ratio of bandwidth over duration
sig_A = np.exp(1j*2*np.pi*(0.5*k*t**2))
# plt.plot(t, np.real(sig_A))
# plt.xlim(0, 0.000010)
# plt.show()

#----Defining the waveform------
tx_waveform = 2**12 * sig_A


rx_time_ms = 1000*len(t)/PlutoSamprate


#--initialize the SDR---
sdr_object = initialize_Pluto_TDD(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms)
print(type(sdr_object))

SDR = sdr_object[1]
TDD = sdr_object[2]

frame_length_samples = m.floor(rx_time_ms*PlutoSamprate/1000)
capture_range = 100

results = pluto_transmit_receive(SDR, TDD, tx_waveform, np.int32(capture_range), np.int32(frame_length_samples),Path)