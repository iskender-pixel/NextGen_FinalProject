import adi
import numpy as np
import scipy.io
import os

#difference for 2 functions: https://chatgpt.com/share/684037a8-ca48-800a-9a62-89dbe3b23f61

def initialize_Pluto_TDD(PlutoIP,sample_rate,center_freq,rx_gain,tx_gain,rx_time_ms):
    
    # %% Setup SDR
    PlutoIP = 'ip:'+PlutoIP
    my_sdr = adi.Pluto(uri=PlutoIP)
    # my_sdr = adi.Pluto()
    tddn = adi.tddn(PlutoIP)
    
    # If you want repeatable alignment between transmit and receive then the rx_lo, tx_lo and sample_rate can only be set once after power up
    # But if you don't care about this, then program sample_rate and LO's as much as you want
    # So after bootup, check if the default sample_rate and LOs are being used, if so then program new ones!
    if (
        30719990 < my_sdr.sample_rate < 30720009
        and 2399999990 < my_sdr.rx_lo < 2400000009
        and 2449999990 < my_sdr.tx_lo < 2450000009
    ):
        my_sdr.sample_rate = int(sample_rate)
        my_sdr.rx_lo = int(center_freq)
        my_sdr.tx_lo = int(center_freq)
        print("Pluto has just booted and I've set the sample rate and LOs!")
    print(f'Sample_rate: {my_sdr.sample_rate}')
    print(f'RX_LO: {my_sdr.rx_lo}')
    print(f'TX_LO: {my_sdr.tx_lo}')


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
    

    frame_length_ms = rx_time_ms

    frame_length_samples = int((rx_time_ms / 1000) * my_sdr.sample_rate)
    N_rx = int(1 * frame_length_samples)
    my_sdr.rx_buffer_size = N_rx

    tddn.startup_delay_ms = 0
    tddn.frame_length_ms = frame_length_ms
    tddn.burst_count = 0  # 0 means repeat indefinitely

    tddn.channel[0].on_raw = 0
    tddn.channel[0].off_raw = 0
    tddn.channel[0].polarity = 1
    tddn.channel[0].enable = 1

    # RX DMA SYNC
    tddn.channel[1].on_raw = 0
    tddn.channel[1].off_raw = 10
    tddn.channel[1].polarity = 0
    tddn.channel[1].enable = 1

    # TX DMA SYNC
    tddn.channel[2].on_raw = 0
    tddn.channel[2].off_raw = 10
    tddn.channel[2].polarity = 0
    tddn.channel[2].enable = 1

    tddn.sync_external = True  # enable external sync trigger
    tddn.enable = True  # enable TDD engine
    
    print("SDR Configuration Completed")

    # Return the my_sdr and tddn objects
    return (my_sdr, tddn)

def initialize_Pluto(PlutoIP,tx_buffer_size,sample_rate,tx_center_freq,rx_center_freq,rx_gain,tx_gain):
    
    # %% Setup SDR
    PlutoIP = 'ip:'+PlutoIP
    my_sdr = adi.Pluto(uri=PlutoIP)
    # my_sdr = adi.Pluto()
    
   
    my_sdr.sample_rate = int(sample_rate)
    
    my_sdr.tx_rf_bandwidth = int(sample_rate)
    my_sdr.rx_rf_bandwidth = int(sample_rate)
    my_sdr.rx_lo = int(tx_center_freq)
    my_sdr.tx_lo = int(rx_center_freq)
        
    print("I've set the sample rate and LOs!")
        
    my_sdr.rx_output_type    
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
    
    frame_length_samples = int(tx_buffer_size)

    my_sdr.rx_buffer_size = frame_length_samples

    print("SDR Configuration Completed")

    # Return the my_sdr and tddn objects
    return (my_sdr)

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

def pluto_transmit_receive(my_sdr, tddn, iq, capture_range, frame_length_samples, path):
    
    # Invia il segnale IQ
    print("sending IQ data")
    
    iq = np.squeeze(iq)  # Removes dimensions of length 1
    
    my_sdr._rx_init_channels()
    my_sdr.tx(iq)
    
    # Enable software trigger for transmission
    # tddn.sync_soft = 1  # Start the TDD transmit
    
    # # Print configuration information
    # print(f"TX/RX Sampling_rate: {my_sdr.sample_rate}")
    # print(f"Number of samples in a frame: {frame_length_samples}")
    # print(f"RX buffer length: {frame_length_samples}")
    # print(f"TX buffer length: {len(iq)}")
    # print(f"RX_receive time[ms]: {((1 / my_sdr.sample_rate) * frame_length_samples) * 1000}")
    # print(f"TX_transmit time[ms]: {((1 / my_sdr.sample_rate) * len(iq)) * 1000}")
    # print(f"TDD_frame time[ms]: {tddn.frame_length_ms}")
    # print(f"TDD_frame time[raw]: {tddn.frame_length_raw}")
    
    # Initialize the array used to store the received data
    received_array = np.zeros(frame_length_samples, dtype=complex) * 1j

    # Receive data
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