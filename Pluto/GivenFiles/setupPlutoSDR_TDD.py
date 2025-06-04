import adi

def initialize_Pluto_TDD(PlutoIP,sample_rate,center_freq,rx_gain,tx_gain,rx_time_ms):
    
    # %% Setup SDR
    PlutoIP = 'ip:'+PlutoIP
    my_sdr = adi.Pluto(uri=PlutoIP)
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