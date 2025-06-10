# -*- coding: utf-8 -*-
"""
Created on Sun May 11 00:37:58 2025

@author: Gaetano
"""

import numpy as np

def continuous_transmit(my_sdr, iq):
    
    # Invia il segnale IQ
    print("sending IQ data")
    
    iq = np.squeeze(iq)  # Removes dimensions of length 1
    
    my_sdr.tx_cyclic_buffer = True  # must be true to use the continuos transmission
    
    my_sdr.tx(iq)
    
    # Print configuration information
    print("TX Transmission Mode: Continuous")
    print(f"TX Sampling_rate: {my_sdr.sample_rate}")
    print(f"TX buffer length: {len(iq)}")
    

def clear_buffer(my_sdr):
    
    # Clear the transmission buffer
    my_sdr.tx_destroy_buffer()
    print("Pluto Buffer Cleared!")
    