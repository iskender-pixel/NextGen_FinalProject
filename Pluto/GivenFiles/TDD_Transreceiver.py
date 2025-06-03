# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 00:37:58 2025

@author: Gaetano
"""

import adi
import numpy as np
import scipy.io
import os

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
    received_array = np.zeros((capture_range, frame_length_samples), dtype=complex) * 1j

    # Receive data
    for r in range(capture_range):
        received_array[r] = my_sdr.rx()

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