import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QDoubleSpinBox, QLabel, QGroupBox, QFormLayout, QSpinBox
import adi
import scipy.io
from threading import Thread
import threading
import scipy.signal
import matplotlib.pyplot as plt



# --- Parameters ---
height, width = 2000, 100
buffer = np.zeros((height, width), dtype=np.float32)
paused = False


# If you want repeatable alignment between transmit and receive then the 
# rx_lo, tx_lo and sample_rate can only be set once after power up. If you
# want to change the following parameters' values after the first run you 
# must reboot the Pluto (disconnect and reconnect it)
Pluto_IP = '192.168.2.1'
        
# Sampling frequency (Hz): Pluto can sample up to 61 MHz but due to the USB 2.0 interface you have to choose values lower 
# or equal to 5MHz if you want to receive 100% of samples over time.
PlutoSamprate = 60.5e6 

# Pluto operating frequency (Hz) must be between 70MHz and 6GHz
centerFrequency = 2.5e9

# -45  % Pluto TX channel Gain must be between 0 and -88  
tx_gain = -20         

# Pluto RX channel Gain must be between -3 and 70   
rx_gain = 40    

settings_lock = threading.Lock()


class SDR():
    def __init__(self, PlutoIP, sample_rate, center_freq, rx_gain, tx_gain, rx_time_ms):
        # %% Setup SDR
        PlutoIP = 'ip:'+PlutoIP
        self.my_sdr = adi.Pluto(uri=PlutoIP)
        self.tddn = adi.tddn(PlutoIP)

        # If you want repeatable alignment between transmit and receive then the rx_lo, tx_lo and sample_rate can only be set once after power up
        # But if you don't care about this, then program sample_rate and LO's as much as you want
        # So after bootup, check if the default sample_rate and LOs are being used, if so then program new ones!
        if (
            30719990 < self.my_sdr.sample_rate < 30720009
            and 2399999990 < self.my_sdr.rx_lo < 2400000009
            and 2449999990 < self.my_sdr.tx_lo < 2450000009
        ):
            self.my_sdr.sample_rate = int(sample_rate)
            self.my_sdr.rx_lo = int(center_freq)
            self.my_sdr.tx_lo = int(center_freq)
            print("Pluto has just booted and I've set the sample rate and LOs!")


        # Configure Rx
        self.my_sdr.rx_enabled_channels = [0]
        sample_rate = int(self.my_sdr.sample_rate)
        # manual or slow_attack
        self.my_sdr.gain_control_mode_chan0 = "manual"
        self.my_sdr.rx_hardwaregain_chan0 = int(rx_gain)
        # Default is 4 Rx buffers are stored, but to immediately see the result, set buffers=1
        self.my_sdr._rxadc.set_kernel_buffers_count(1)

        # Configure Tx
        self.my_sdr.tx_enabled_channels = [0]
        self.my_sdr.tx_hardwaregain_chan0 = int(tx_gain)
        self.my_sdr.tx_cyclic_buffer = True  # must be true to use the TDD transmit


        frame_length_ms = rx_time_ms

        frame_length_samples = int((rx_time_ms / 1000) * self.my_sdr.sample_rate)
        N_rx = int(1 * frame_length_samples)
        self.my_sdr.rx_buffer_size = N_rx

        self.tddn.startup_delay_ms = 0
        self.tddn.frame_length_ms = frame_length_ms
        self.tddn.burst_count = 0  # 0 means repeat indefinitely

        self.tddn.channel[0].on_raw = 0
        self.tddn.channel[0].off_raw = 0
        self.tddn.channel[0].polarity = 1
        self.tddn.channel[0].enable = 1

        # RX DMA SYNC
        self.tddn.channel[1].on_raw = 0
        self.tddn.channel[1].off_raw = 10
        self.tddn.channel[1].polarity = 0
        self.tddn.channel[1].enable = 1

        # TX DMA SYNC
        self.tddn.channel[2].on_raw = 0
        self.tddn.channel[2].off_raw = 10
        self.tddn.channel[2].polarity = 0
        self.tddn.channel[2].enable = 1

        self.tddn.sync_external = True  # enable external sync trigger
        self.tddn.enable = True  # enable TDD engine

        print("SDR Configuration Completed")

    def __del__(self):
        pass

    def transmit_receive(self, iq, capture_range, frame_length_samples):
        
        # Invia il segnale IQ
        print("sending IQ data")

        iq = np.squeeze(iq)  # Removes dimensions of length 1

        self.my_sdr._rx_init_channels()
        self.my_sdr.tx(iq)

        # Enable software trigger for transmission
        self.tddn.sync_soft = 1  # Start the TDD transmit

        # Print configuration information
        print(f"TX/RX Sampling_rate: {self.my_sdr.sample_rate}")
        print(f"Number of samples in a frame: {frame_length_samples}")
        print(f"RX buffer length: {frame_length_samples}")
        print(f"TX buffer length: {len(iq)}")
        print(f"RX_receive time[ms]: {((1 / self.my_sdr.sample_rate) * frame_length_samples) * 1000}")
        print(f"TX_transmit time[ms]: {((1 / self.my_sdr.sample_rate) * len(iq)) * 1000}")
        print(f"TDD_frame time[ms]: {self.tddn.frame_length_ms}")
        print(f"TDD_frame time[raw]: {self.tddn.frame_length_raw}")

        # Initialize the array used to store the received data
        received_array = np.zeros((capture_range, frame_length_samples), dtype=complex) * 1j

        # Receive data
        for r in range(capture_range):
            received_array[r] = self.my_sdr.rx()

        # Shutdown Pluto transmission
        self.tddn.enable = 0
        for i in range(3):
            self.tddn.channel[i].on_ms = 0
            self.tddn.channel[i].off_raw = 0
            self.tddn.channel[i].polarity = 0
            self.tddn.channel[i].enable = 1
        self.tddn.enable = 1
        self.tddn.enable = 0

        # Clear the transmission buffer
        self.my_sdr.tx_destroy_buffer()
        print("Pluto Buffer Cleared!")

        # Save
        # Should we want it, here we can have the program dump a save of whatever it received.

        # Return the received_array list
        return received_array  
    
class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.parent = parent
        self.current_settings = parent.settings.copy()
        self.new_settings = parent.settings.copy()

        layout = QVBoxLayout(self)
       
        chirp_group = QGroupBox("Chirp Settings")
        chirp_layout = QFormLayout()
        self.spin_box1 = QDoubleSpinBox()
        self.spin_box2 = QDoubleSpinBox()

        self.spin_box1.setRange(0, 1000.0)
        self.spin_box1.setDecimals(1)
        self.spin_box1.setSingleStep(0.1)
        self.spin_box1.setValue(self.current_settings["chirp_duration"] * 1e6)

        self.spin_box2.setRange(0, 30.0)
        self.spin_box2.setDecimals(1)
        self.spin_box2.setSingleStep(0.1)
        self.spin_box2.setValue(self.current_settings["chirp_bandwidth"] /1e6)

        chirp_layout.addRow("Chirp duration [us]:", self.spin_box1)
        chirp_layout.addRow("Chirp bandwidth [MHz]:", self.spin_box2)
        chirp_group.setLayout(chirp_layout)



        filter_group = QGroupBox("Filter Settings")
        filter_layout = QFormLayout()
        self.spin_box3 = QSpinBox()
        self.spin_box4 = QDoubleSpinBox()

        self.spin_box3.setRange(1, 10)
        self.spin_box3.setSingleStep(1)
        self.spin_box3.setValue(self.current_settings["filter_order"])

        self.spin_box4.setRange(0, 15.0)
        self.spin_box4.setDecimals(1)
        self.spin_box4.setSingleStep(0.1)
        self.spin_box4.setValue(self.current_settings["filter_bandwidth"] /1e6)

        filter_layout.addRow("Filter order:", self.spin_box3)
        filter_layout.addRow("Filter bandwith [MHz]:", self.spin_box4)
        filter_group.setLayout(filter_layout)



        other_group = QGroupBox("Other Settings")
        other_layout = QFormLayout()
        self.spin_box5 = QSpinBox()

        self.spin_box5.setRange(100, 10000)
        self.spin_box5.setSingleStep(100)
        self.spin_box5.setValue(int(self.current_settings["sleep_time"]*1e3))

        other_layout.addRow("Sleep time [ms]:", self.spin_box5)
        other_group.setLayout(other_layout)

        btn_apply = QPushButton("Apply")
        btn_default = QPushButton("Restore default")
        btn_cancel = QPushButton("Cancel")

        layout.addWidget(chirp_group)
        layout.addWidget(filter_group)
        layout.addWidget(other_group)
        layout.addWidget(btn_apply)
        layout.addWidget(btn_default)
        layout.addWidget(btn_cancel)

        btn_apply.clicked.connect(self.apply)
        btn_default.clicked.connect(self.default)
        btn_cancel.clicked.connect(self.cancel)

    def apply(self):
        self.new_settings["chirp_duration"] = self.spin_box1.value() * 1e-6  # convert back to seconds
        self.new_settings["chirp_bandwidth"] = self.spin_box2.value() * 1e6
        self.new_settings["filter_order"] = self.spin_box3.value()
        self.new_settings["filter_bandwidth"] = self.spin_box4.value() * 1e6
        self.new_settings["sleep_time"] = self.spin_box5.value() * 1e-3

        self.accept()  # Use accept() to close dialog and mark as accepted

    def default(self):
        self.new_settings = self.parent.default_settings.copy()
        self.accept() 

    def cancel(self):
        self.close()  

    def get_settings(self):
        self.close()  
        return self.new_settings


class ImageUpdater(QtWidgets.QWidget):
    update_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, application):
        super().__init__()

        self.settings_changed = True
        self.app = application
        self.save_index = 0
        self.settings = {
            "chirp_duration" : 100e-6,
            "chirp_bandwidth" : 15e6,
            "sleep_time" : 0.1,
            "filter_bandwidth" : 1E6,
            "filter_order" : 1,
            "chirp_count" : 100
            }
        
        self.default_settings = self.settings.copy()

        # --- Layout Setup ---
        self.setWindowTitle("PyQt6 + PyQtGraph External Update Plot")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)

        # --- Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Buffer")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.settings_button = QtWidgets.QPushButton("Settings")
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.settings_button)

        layout.addLayout(button_layout)

        # --- Plot ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot = self.plot_widget.addPlot()
        self.img_item = pg.ImageItem()

        self.plot.addItem(self.img_item)

        # Create inferno colormap using matplotlib
        inferno_cmap = plt.get_cmap("inferno")
        lut = (inferno_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        # Apply lookup table to the image
        self.img_item.setLookupTable(lut)




        self.cm = pg.colormap.get('inferno') # prepare a linear color map
        self.img_item.setColorMap(self.cm)


        self.plot.setRange(xRange=(0, width), yRange=(0, height))
        self.plot.disableAutoRange()
        self.plot.vb.setMouseEnabled(x=False, y=False)
        self.plot.invertY(False)

        layout.addWidget(self.plot_widget)

        # --- State ---
        self.buffer = np.zeros((height, width), dtype=np.float32)
        self.img_item.setImage(self.buffer, autoLevels=False, levels=(0, 1))
        self.paused = False

        # --- Button Callbacks ---
        self.save_button.clicked.connect(self.save_buffer)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.settings_button.clicked.connect(self.open_settings)


        # --- Signal for thread-safe updates ---
        self.update_signal.connect(self.update_image)

    def update_image(self, new_col: np.ndarray):
        if self.paused:
            return
        
        if not self.app.instance():
            return  # QApplication is gone

        if new_col.shape != (height,):
            raise ValueError(f"Expected shape ({height},), got {new_col.shape}")

        self.buffer[:, 1:] = self.buffer[:, :-1]
        self.buffer[:, 0] = new_col
        self.img_item.setImage(self.buffer.T, autoLevels=True)

    def save_buffer(self):
        self.save_index += 1
        print("Saving buffer... shape:", self.buffer.shape)
        np.save(f"buffer_{self.save_index}.npy", self.buffer)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def open_settings(self):

        with settings_lock:
            dialog = SettingsDialog(self)
            if dialog.exec():  # returns True if 'Apply' clicked
                self.settings = dialog.get_settings()
                self.settings_changed = True



def obtain_radar_data(updater: ImageUpdater):
    def run():       
        # Create initial chirp
        t = np.arange(0, updater.settings["chirp_duration"], 1/PlutoSamprate)

        rx_time_ms = 1000*len(t)/PlutoSamprate
        chirp = (2**12) * scipy.signal.chirp(t, 0, updater.settings["chirp_duration"], updater.settings["chirp_bandwidth"], complex=True)
        chirp = np.tile(chirp, updater.settings["chirp_count"])

        pluto = SDR(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms)   
        b, a = scipy.signal.butter(1, Wn=updater.settings["filter_bandwidth"], btype='lowpass', fs=PlutoSamprate)      

        while True:
            with settings_lock:
                if updater.settings_changed:
                    t = np.arange(0,updater.settings["chirp_duration"], 1/PlutoSamprate) 
                    chirp = (2**12) * scipy.signal.chirp(t, 0, updater.settings["chirp_duration"], updater.settings["chirp_bandwidth"], complex=True)
                    rx_time_ms = 1000*len(t)/PlutoSamprate
                    #pluto = SDR(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms)   
                    b, a = scipy.signal.butter(updater.settings["filter_order"], Wn=updater.settings["filter_bandwidth"], btype='lowpass', fs=PlutoSamprate) 
                    sleep_time = updater.settings["sleep_time"]
                    updater.settings_changed = False

                #frame_length_samples = int(np.floor(rx_time_ms*PlutoSamprate/1000))
                #capture_range = 100

                #results = pluto.transmit_receive(iq=chirp, capture_range=capture_range, frame_length_samples=frame_length_samples)

                #rx = np.sum(results, 0)

                #tx = 100 * chirp
                #beat_signal = tx * np.conj(rx)
                #
                #
                #filtered = scipy.signal.filtfilt(b, a, beat_signal)

                #filtered_fft = np.fft.fftshift(np.fft.fft(filtered))
                #new_col = filtered_fft[int(filtered_fft.shape[0]/2):]


                y = np.linspace(0, 4 * np.pi, height)
                new_col =  chirp[0:2000] #0.5 + 0.5 * np.sin(y + time.time()) + 0.1 * np.random.randn(height) 
                try: 
                    updater.update_signal.emit(new_col.astype(np.float32))
                except RuntimeError:
                    pass
            time.sleep(sleep_time)
    updater.thread = Thread(target=run, daemon=True).start()


# --- Run ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    updater = ImageUpdater(app)
    updater.show()
    obtain_radar_data(updater)
    sys.exit(app.exec())
