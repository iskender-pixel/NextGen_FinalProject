# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 00:37:58 2025

@author: Gaetano
"""


import numpy as np
import scipy.io
import os

def receive_data(my_sdr, frame_length_samples, path):
    
    my_sdr._rx_init_channels()
    
    # Print configuration information
    print(f"RX Sampling_rate: {my_sdr.sample_rate}")
    print(f"Number of samples in a frame: {frame_length_samples}")
    print(f"RX buffer length: {frame_length_samples}")
    
    # Initialize the array used to store the received data
    received_array = np.zeros((1, frame_length_samples), dtype=complex)

    
    received_array = my_sdr.rx()

    # Save
    mat_file_path = os.path.join(path, 'received_data.mat')
    scipy.io.savemat(mat_file_path, {'received_data': received_array})

    print(f"Data saved to {mat_file_path}")
    
    # Return the received_array list
    return received_array.tolist()