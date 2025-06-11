import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QDoubleSpinBox, QLabel, QGroupBox, QFormLayout, QSpinBox, QAbstractSpinBox, QMessageBox
import adi
import scipy.io
from threading import Thread
import threading
import scipy.signal
import matplotlib.pyplot as plt



# --- Parameters ---
height, width = 3025, 100
buffer = np.zeros((height, width))
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

settings_lock = threading.RLock()


class SDR():
    def __init__(self, PlutoIP, sample_rate, center_freq, rx_gain, tx_gain, rx_time_ms, N):
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

        frame_length_samples = N
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

        #print("SDR Configuration Completed")

    def __del__(self):
        pass

    def transmit_receive(self, iq, capture_range, frame_length_samples):
        
        # Invia il segnale IQ
        #print("sending IQ data")

        iq = np.squeeze(iq)  # Removes dimensions of length 1

        self.my_sdr._rx_init_channels()
        self.my_sdr.tx(iq)

        # Enable software trigger for transmission
        self.tddn.sync_soft = 1  # Start the TDD transmit

        # Print configuration information
        #print(f"TX/RX Sampling_rate: {self.my_sdr.sample_rate}")
        #print(f"Number of samples in a frame: {frame_length_samples}")
        #print(f"RX buffer length: {frame_length_samples}")
        #print(f"TX buffer length: {len(iq)}")
        #print(f"RX_receive time[ms]: {((1 / self.my_sdr.sample_rate) * frame_length_samples) * 1000}")
        #print(f"TX_transmit time[ms]: {((1 / self.my_sdr.sample_rate) * len(iq)) * 1000}")
        #print(f"TDD_frame time[ms]: {self.tddn.frame_length_ms}")
        #print(f"TDD_frame time[raw]: {self.tddn.frame_length_raw}")

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
        #print("Pluto Buffer Cleared!")

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
        self.spin_box6 = QSpinBox()

        self.spin_box1.setRange(1, 1000.0)
        self.spin_box1.setDecimals(1)
        self.spin_box1.setSingleStep(1)
        self.spin_box1.setValue(self.current_settings["chirp_duration"] * 1e6)
        self.spin_box1.setToolTip("Change the duration of the chirp. Measured in microseconds")

        self.spin_box2.setRange(0.1, 30.0)
        self.spin_box2.setDecimals(1)
        self.spin_box2.setSingleStep(0.1)
        self.spin_box2.setValue(self.current_settings["chirp_bandwidth"] /1e6)
        self.spin_box2.setToolTip("Change the bandwidth of the chirp. Measured in megahertz")

        self.spin_box6.setRange(1, 500)
        self.spin_box6.setSingleStep(10)
        self.spin_box6.setValue(self.current_settings["chirp_count"])
        self.spin_box6.setToolTip("Change the amount of chirps per chirp train.")

        chirp_layout.addRow("Chirp duration [us]:", self.spin_box1)
        chirp_layout.addRow("Chirp bandwidth [MHz]:", self.spin_box2)
        chirp_layout.addRow("Chirp count:", self.spin_box6)
        chirp_group.setLayout(chirp_layout)


        filter_group = QGroupBox("Filter Settings")
        filter_layout = QFormLayout()
        self.spin_box3 = QSpinBox()
        self.spin_box4 = QDoubleSpinBox()

        self.spin_box3.setRange(1, 10)
        self.spin_box3.setSingleStep(1)
        self.spin_box3.setValue(self.current_settings["filter_order"])
        self.spin_box3.setToolTip("Set the order of the filter. This filter is applied after mixing to remove the high image.")

        self.spin_box4.setRange(0.1, 15.0)
        self.spin_box4.setDecimals(1)
        self.spin_box4.setSingleStep(0.1)
        self.spin_box4.setValue(self.current_settings["filter_bandwidth"] /1e6)
        self.spin_box4.setToolTip("Change the bandwidth of the filter. This filter is applied after mixing to remove the high image.")

        filter_layout.addRow("Filter order:", self.spin_box3)
        filter_layout.addRow("Filter bandwith [MHz]:", self.spin_box4)
        filter_group.setLayout(filter_layout)

        fft_group = QGroupBox("FFT Settings")
        fft_layout = QFormLayout()
        self.spin_box7 = QSpinBox()
        self.spin_box8 = QDoubleSpinBox()

        self.spin_box7.setRange(1, 250)
        self.spin_box7.setSingleStep(1)
        self.spin_box7.setValue(self.current_settings["zero_padding_factor"])
        self.spin_box7.setToolTip("Change the amount of zero padding used in the FFT. A setting of five equals to five times the lenght of the received sample.")

        self.spin_box8.setRange(0.1, 250.0)
        self.spin_box8.setDecimals(1)
        self.spin_box8.setSingleStep(0.1)
        self.spin_box8.setValue(self.current_settings["maximum_distance"])
        self.spin_box8.setToolTip("Change the maximum distance displayed on the viewport. The viewport will try to match this distance as best it is able to.")

        fft_layout.addRow("Zero padding factor:", self.spin_box7)
        fft_layout.addRow("Maximum distance:", self.spin_box8)
        fft_group.setLayout(fft_layout)



        other_group = QGroupBox("Other Settings")
        other_layout = QFormLayout()
        self.spin_box5 = QSpinBox()
        self.check_box1 = QtWidgets.QCheckBox()

        self.spin_box5.setRange(100, 10000)
        self.spin_box5.setSingleStep(100)
        self.spin_box5.setValue(int(self.current_settings["sleep_time"]*1e3))
        self.spin_box5.setToolTip("Change the sleep time of the program. This is the time between sending chirp trains. Time taken to process results is taken into account as best as the program is able to ensure consistent timing.")

        self.check_box1.setChecked(self.current_settings["debug"])
        self.check_box1.setToolTip("If this is turned on, the input of the array is a noisy sin wave, and the enviromental correction is a linear sweep")

        other_layout.addRow("Sleep time [ms]:", self.spin_box5)
        other_layout.addRow("Debug mode:", self.check_box1)
        other_group.setLayout(other_layout)

        btn_apply = QPushButton("Apply")
        btn_default = QPushButton("Restore default")
        btn_cancel = QPushButton("Cancel")

        layout.addWidget(chirp_group)
        layout.addWidget(filter_group)
        layout.addWidget(fft_group)
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
        self.new_settings["chirp_count"] = self.spin_box6.value()
        self.new_settings["zero_padding_factor"] = self.spin_box7.value()
        self.new_settings["maximum_distance"] = self.spin_box8.value()
        self.new_settings["debug"] = self.check_box1.isChecked()

        self.accept()  # Use accept() to close dialog and mark as accepted

    def default(self):
        self.new_settings = self.parent.default_settings.copy()
        self.accept() 

    def cancel(self):
        self.close()  

    def get_settings(self):
        self.close()  
        return self.new_settings
    
class CallibrationDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Environment Calibration Window")
        self.parent = parent
        
        self.new_enviro_calibration = parent.enviro_calibration.copy()
        self.current_enviro_callibration = parent.enviro_calibration.copy()

        layout = QVBoxLayout(self)
       
        cal_group = QGroupBox("Calibration Settings")
        cal_layout = QFormLayout()
        self.spin_box1 = QSpinBox()
        self.label = QtWidgets.QLabel()
        if np.abs(np.sum(self.new_enviro_calibration[10])) == 0.0:
            self.label.setText(f"Calibration for this angle has <span style='color:red'>not been saved</span>")
        else:
            self.label.setText(f"Calibration for this angle has <span style='color:green'>been saved</span>")

        self.push_button1 = QtWidgets.QPushButton("Calibrate for this angle")
        self.push_button2 = QtWidgets.QPushButton("Save the calibration")
        self.push_button3 = QtWidgets.QPushButton("Load a calibration")

        self.spin_box1.setRange(-20,20)
        self.spin_box1.setSingleStep(2)
        self.spin_box1.setValue(0)
        self.spin_box1.lineEdit().setReadOnly(True)
        self.spin_box1.setPrefix("Angle: ")
        self.spin_box1.setSuffix("°")
        self.spin_box1.setToolTip("Change the angle for which to record calibration data for.")

        self.push_button1.setToolTip("Perform callibration for the angle specified above. A window will open with instructions.")
        self.push_button2.setToolTip("Save the current callibration settings. If callibration was just performed, apply the callibration first then reopen this window to save that data.")
        self.push_button3.setToolTip("Load callibration data.")

        cal_layout.addWidget(self.spin_box1)
        cal_layout.addWidget(self.label)
        cal_layout.addWidget(self.push_button1)
        cal_layout.addWidget(self.push_button2)
        cal_layout.addWidget(self.push_button3)
        cal_group.setLayout(cal_layout)
        

        btn_apply = QPushButton("Accept")
        btn_cancel = QPushButton("Cancel")

        layout.addWidget(cal_group)
        layout.addWidget(btn_apply)
        layout.addWidget(btn_cancel)

        btn_apply.clicked.connect(self.apply)
        btn_cancel.clicked.connect(self.cancel)
        self.push_button1.clicked.connect(self.calibrate)
        self.push_button2.clicked.connect(self.save)
        self.push_button3.clicked.connect(self.load)
        self.spin_box1.valueChanged.connect(self.update_label)


    def update_label(self):
        ind = int((self.spin_box1.value() + 20 ) / 2)
        if np.abs(np.sum(self.new_enviro_calibration[ind])) == 0.0:
            self.label.setText(f"Calibration for this angle has <span style='color:red'>not been saved</span>")
        else:
            self.label.setText(f"Calibration for this angle has <span style='color:green'>been saved</span>")


    def apply(self):
        self.accept()

    def cancel(self):
        self.new_enviro_calibration = self.current_enviro_callibration
        self.close()  

    def save(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, "Save Current Calibration Data (not including changes)", "/home/jana", "Numpy Files (*.npy)")    
        if fileName[0] != '':
            np.save(fileName[0], self.current_enviro_callibration)  

    def load(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Load Calibration Data", "/home/jana", "Numpy Files (*.npy)")
        if fileName[0] != '':
            self.new_enviro_calibration = np.load(fileName[0]) 
            self.update_label()
            
    def calibrate(self):
        angle = self.spin_box1.value()
        angle_ind = int((angle + 20 ) / 2)

        msgBox = QMessageBox()
        msgBox.setText("Please Ensure the area to be calibrated for is clear and still.")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        ret = msgBox.exec()
        observed_values = None
        if (ret == QMessageBox.StandardButton.Ok):
            for i in range(5):
                t = np.arange(0,self.parent.settings["chirp_duration"], 1/PlutoSamprate)
                if not self.parent.settings["debug"]: 
                # Recalculate the chirp
                    T = updater.settings["chirp_duration"]
                    B = updater.settings["chirp_bandwidth"]
                    fs = PlutoSamprate

                    t_triangle1 = np.arange(0, T / 2 - 1 / fs, 1 / fs)
                    t_triangle2 = np.flip(t_triangle1)

                    k = B / T  # Chirp slope defined as ratio of bandwidth over duration
                    Chirp1 = np.exp(1j * 2 * np.pi * (k * t_triangle1 ** 2))
                    Chirp2 = np.exp(1j * 2 * np.pi * (k * t_triangle2 ** 2))

                    chirp = (2 ** 12) * np.append(Chirp1[:-1], Chirp2)
                    rx_time_ms = 1000*len(t)/PlutoSamprate
                    frame_length_samples = int(np.floor(rx_time_ms*PlutoSamprate/1000))
                    pluto = SDR(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms, frame_length_samples)   
                    results = pluto.transmit_receive(iq=chirp, capture_range=self.parent.settings["chirp_count"], frame_length_samples=frame_length_samples)
                    rx = np.sum(results, 0)
                    tx = 100 * chirp
                    beat_signal = tx * np.conj(rx)     
                    b, a = scipy.signal.butter(1, Wn=self.parent.settings["filter_bandwidth"], btype='lowpass', fs=PlutoSamprate)      
                    filtered = scipy.signal.filtfilt(b, a, beat_signal)
                    filtered_fft = np.fft.fft(filtered, n=65536)#len(filtered)*self.parent.settings["zero_padding_factor"])
                    if observed_values is None:
                        observed_values = filtered_fft
                    else:
                       observed_values = np.add(observed_values, filtered_fft)
                time.sleep(0.2)
            observed_values = observed_values / 5

            msgBox = QMessageBox()
            msgBox.setText("Callibration complete.")
            msgBox.exec()   

        self.new_enviro_calibration = observed_values
        self.update_label()
        
    def get_calibration(self):
        self.close()
        return self.new_enviro_calibration


class ImageUpdater(QtWidgets.QWidget):
    global settings_lock
    update_signal = QtCore.pyqtSignal(tuple)
    error_signal = QtCore.pyqtSignal(bool) # Could upgrade this to also share which type of error occured, so that the user can be informed of multiple kinds of errors

    def __init__(self, application):
        super().__init__()

        self.settings_changed = True
        self.app = application
        self.save_index = 0
        self.plot_height = 100
        self.plot_width = 100
        self.callibration_offset = 223
        self.cal_hard_enabled = False
        self.cal_environ_enabled = False
        self.distances = None
        self.cleared = False

        self.new_col = None
        self.freq = None

        self.settings = {
            "chirp_duration" : 200e-6,
            "chirp_bandwidth" : 30e6,
            "sleep_time" : 1,
            "filter_bandwidth" : 1E6,
            "filter_order" : 1,
            "chirp_count" : 100,
            "maximum_distance": 15,
            "zero_padding_factor": 200,
            "debug" : False
            }
        
        self.default_settings = self.settings.copy()
        

        # --- Layout Setup ---
        self.setWindowTitle("Radar Control Panel")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)
        

        # --- Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Buffer")
        self.clear_buffer = QtWidgets.QPushButton("Clear buffer")
        self.new_button = QtWidgets.QPushButton("Aquire new")
        self.settings_button = QtWidgets.QPushButton("Settings")

        self.save_button.setToolTip("Save the current buffer that is displayed in the viewport for later viewing.")
        self.clear_buffer.setToolTip("Clear the buffer. This also pauses the program. The primary purpose of this feature is to preview environment calibration data.")
        self.new_button.setToolTip("Send out a new train of chirps, then process and display the result.")
        self.settings_button.setToolTip("Open the settings menu.")
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_buffer)
        button_layout.addWidget(self.new_button)
        button_layout.addWidget(self.settings_button)

        layout.addLayout(button_layout)

        button_layout2 = QtWidgets.QHBoxLayout()
        self.cal_offset = QtWidgets.QPushButton("Calibrate hardware offset")
        self.cal_offset.setToolTip("Calibrate for hardware induced offset. A window with instructions will open.")
        self.cal_enviro = QtWidgets.QPushButton("Calibrate environment")
        self.cal_enviro.setToolTip("Calibrate for a static environment to reduce its influence.")
        self.use_cal_hard = QtWidgets.QCheckBox("Enable harware calibration")
        self.use_cal_hard.setToolTip("Enable hardware calibration if it is available.")
        self.use_cal_hard.setChecked(False)

        self.use_cal_environ = QtWidgets.QCheckBox("Enable environment calibration")
        self.use_cal_environ.setToolTip("Enable environment calibration if it is available.")

        self.cal_angle = QSpinBox()
        self.cal_angle.setValue(0)
        self.cal_angle.setRange(-20,20)
        self.cal_angle.setPrefix("Angle: ")
        self.cal_angle.setSuffix("°")
        self.cal_angle.lineEdit().setReadOnly(True)
        self.cal_angle.setSingleStep(2)

        button_layout2.addWidget(self.cal_offset)
        button_layout2.addWidget(self.use_cal_hard)
        button_layout2.addWidget(self.cal_enviro)
        button_layout2.addWidget(self.cal_angle)
        button_layout2.addWidget(self.use_cal_environ)

        layout.addLayout(button_layout2)


        # --- Plot ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_item = pg.PlotItem()        

        self.plot_item.showGrid(x=True, y=True, alpha=0.3)

        self.plot_widget.addItem(self.plot_item)
        self.axis = self.plot_item.getAxis('bottom')

        self.plot_item.setRange(xRange=(0, 300), yRange=(-3, 1))
        self.plot_item.disableAutoRange()
        self.plot_item.vb.setMouseEnabled(x=False, y=False)
        self.plot_item.invertY(False)

        layout.addWidget(self.plot_widget)

        # --- State ---
        self.enviro_calibration = np.zeros(65536)
        self.paused = False

        # --- Button Callbacks ---
        self.save_button.clicked.connect(self.save_buffer)
        self.clear_buffer.clicked.connect(self.buff_clear)

        self.new_button.clicked.connect(self.aquire_new)
        self.settings_button.clicked.connect(self.open_settings)

        self.cal_offset.clicked.connect(self.cal_hardware)
        self.cal_enviro.clicked.connect(self.cal_environ)
        self.cal_angle.valueChanged.connect(self.cal_angle_update)

        self.use_cal_hard.stateChanged.connect(self.cal_hard_enable)
        self.use_cal_environ.stateChanged.connect(self.cal_environ_enable)

        # --- Signal for thread-safe updates ---
        self.update_signal.connect(self.update_plot)
        self.error_signal.connect(self.error_popup)

    def update_plot(self, tup):
        if self.paused:
            pass
        else:
            self.cleared = False
        new_col, freq = tup

        self.new_col = new_col.copy()
        self.freq = freq.copy()

        if self.paused or not self.app.instance():
            return

        if self.cal_hard_enabled:
            freq = freq - self.callibration_offset

        self.plot_item.clear()
        if self.cal_environ_enabled:
            self.plot_item.plot(freq, np.log10(np.abs(new_col)) - np.log10(np.abs(self.enviro_calibration)))
        else:
            self.plot_item.plot(freq, np.square(np.square(np.abs(new_col))))

    def buff_clear(self):
        global settings_lock
        #TODO re-implement
        with settings_lock:
            self.paused = True
            self.pause_button.setText("Resume" if self.paused else "Pause")
            self.resize_buffer(self.buffer.shape[0]) 
            if self.cal_environ_enabled:
                self.img_item.setImage(np.abs(self.buffer.T - self.enviro_calibration_tiled.T), autoLevels=True)
            else:
                self.img_item.setImage(np.abs(self.buffer.T), autoLevels=True)
            self.cleared = True

    def resize_buffer(self, buf_height):
        #TODO decide if still necessary?
        new_buffer = np.zeros((buf_height, self.plot_width), dtype=np.complex128)
        self.buffer = new_buffer
        print(f"New buffer size: {new_buffer.shape}")

        #new_buffer2 = np.zeros((buf_height, 21), dtype=np.complex128)
        #self.enviro_calibration = new_buffer2

        new_buffer3 = np.zeros((buf_height, self.plot_width), dtype=np.complex128)
        self.enviro_calibration_tiled = new_buffer3
        self.cal_angle_update()
        
    def save_buffer(self):
        self.save_index += 1
        print("Saving buffer... shape:", self.buffer.shape)
        np.save(f"buffer_{self.save_index}_N=65536.npy", self.new_col)
        np.save(f"distances_{self.save_index}_N=65536.npy", self.freq)


    def load_buffer(self):
        pass #TODO implement load buffer function, which should pop up a loaded buffer for inspection

    def aquire_new(self):
        obtain_radar_data(self)

    def open_settings(self):
        with settings_lock:
            dialog = SettingsDialog(self)
            if dialog.exec():  # returns True if 'Apply' clicked
                self.settings = dialog.get_settings()
                self.plot_item.setRange(xRange=(0, self.settings["maximum_distance"]), yRange=(0, 1))
                self.update_plot((self.new_col, self.freq))

    def cal_environ(self):
        with settings_lock:
            dialog = CallibrationDialog(self)
            if dialog.exec():  # returns True if 'Apply' clicked
                pass
                self.enviro_calibration = dialog.get_calibration()
    
    def cal_hardware(self):
        global settings_lock
        with settings_lock:
            msgBox = QMessageBox()
            msgBox.setText("Please connect the short loop.")
            msgBox.setInformativeText("Connect a short cable between the input and output ports of the SDR or radar system. Click ok once this is done. The program will then determine the delay between the input and output.")
            msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            ret = msgBox.exec()

            observed_values = []
            if (ret == QMessageBox.StandardButton.Ok):
                for i in range(5):
                    t = np.arange(0,self.settings["chirp_duration"], 1/PlutoSamprate) 
                    if self.settings["debug"]:
                        observed_values.append(222.55555)
                    else:
                        # Recalculate the chirp
                        T = updater.settings["chirp_duration"]
                        B = updater.settings["chirp_bandwidth"]
                        fs = PlutoSamprate

                        t_triangle1 = np.arange(0, T / 2 - 1 / fs, 1 / fs)
                        t_triangle2 = np.flip(t_triangle1)

                        k = B / T  # Chirp slope defined as ratio of bandwidth over duration
                        Chirp1 = np.exp(1j * 2 * np.pi * (k * t_triangle1 ** 2))
                        Chirp2 = np.exp(1j * 2 * np.pi * (k * t_triangle2 ** 2))

                        chirp = (2 ** 12) * np.append(Chirp1[:-1], Chirp2)


                        rx_time_ms = 1000*len(t)/PlutoSamprate
                        frame_length_samples = int(np.floor(rx_time_ms*PlutoSamprate/1000))
                        pluto = SDR(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms, frame_length_samples)   
                        results = pluto.transmit_receive(iq=chirp, capture_range=self.settings["chirp_count"], frame_length_samples=frame_length_samples)
                        rx = np.sum(results, 0)
                        tx = 100 * chirp
                        beat_signal = tx * np.conj(rx)     
                        b, a = scipy.signal.butter(1, Wn=self.settings["filter_bandwidth"], btype='lowpass', fs=PlutoSamprate)      
                        filtered = scipy.signal.filtfilt(b, a, beat_signal)
                        filtered_fft = np.fft.fft(filtered, n=len(filtered)*self.settings["zero_padding_factor"])
                        distance = self.distances[np.argmax(np.abs(filtered_fft))]
                        observed_values.append(distance)
                    time.sleep(0.2)

                rounded_values = [round(val, 2) for val in observed_values]
                avg = np.average(np.array(observed_values)) 

                msgBox = QMessageBox()
                msgBox.setText("Do you want to use this value?")
                msgBox.setInformativeText(f"The observed values were {rounded_values}. The average value was {avg:.2f}. This value will be used to calibrate the received data. If any chirp related settings are changed, it is recommend to redo callibration.")
                msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                ret = msgBox.exec()   

                if (ret == QMessageBox.StandardButton.Ok):
                    self.callibration_offset = avg
                    if self.cal_hard_enabled:
                        self.update_plot((self.new_col, self.freq))

    def cal_angle_update(self):
        global settings_lock
        if self.cal_environ_enabled:
            with settings_lock:
                angle = self.cal_angle.value()
                angle_ind = int((angle + 20) / 2)

                arr = None
                if self.settings["debug"]:
                    if angle_ind == 10:
                        arr = np.arange(0,self.buffer.shape[0])/ (0.05 * self.buffer.shape[0])
                    elif angle_ind < 10:
                        arr = np.flip(np.arange(0,self.buffer.shape[0])/ (0.1 * self.buffer.shape[0]))
                    else:
                        arr = np.flip(np.geomspace(0.01,1,self.buffer.shape[0]))
                else:
                    arr = self.enviro_calibration.copy()
                arr = arr.reshape((arr.shape[0],1))
                self.enviro_calibration_tiled = np.tile(arr, self.plot_width)
                if self.cleared:
                    self.img_item.setImage(np.abs(self.buffer.T - self.enviro_calibration_tiled.T), autoLevels=True)

    def cal_hard_enable(self, value):
        with settings_lock:
            self.cal_hard_enabled = value
            self.update_plot((self.new_col, self.freq))

    def cal_environ_enable(self, value):
        global settings_lock
        with settings_lock:
            self.cal_environ_enabled = value
            self.update_plot((self.new_col, self.freq))

    
    def error_popup(self, value):
        if value:
            msgBox = QMessageBox()
            msgBox.setText("An error has occured")
            # Here I could upgrade the function to show a different message based on the error
            msgBox.setInformativeText(f"The program was unable to find the SDR. The program has been switched to debug mode. Please connect the SDR, then disable the debug mode from the settings menu to try again.")
            msgBox.exec()

def obtain_radar_data(updater: ImageUpdater):
    global settings_lock
    def run():       
        # Create initial chirp
        t = np.arange(0, updater.settings["chirp_duration"], 1/PlutoSamprate)

        rx_time_ms = 1000*len(t)/PlutoSamprate

        T = updater.settings["chirp_duration"]
        B = updater.settings["chirp_bandwidth"]
        fs = PlutoSamprate

        t_triangle1 = np.arange(0, T / 2 - 1 / fs, 1 / fs)
        t_triangle2 = np.flip(t_triangle1)

        k = B / T  # Chirp slope defined as ratio of bandwidth over duration
        Chirp1 = np.exp(1j * 2 * np.pi * (k * t_triangle1 ** 2))
        Chirp2 = np.exp(1j * 2 * np.pi * (k * t_triangle2 ** 2))

        chirp = (2**12) * np.append(Chirp1[:-1], Chirp2)

        frame_length_samples = int(np.floor(rx_time_ms*PlutoSamprate/1000))
        zero_padding = updater.settings["zero_padding_factor"]
        updater.resize_buffer(np.min([int((frame_length_samples * zero_padding) / 8), 5000]))
        b, a = scipy.signal.butter(1, Wn=updater.settings["filter_bandwidth"], btype='lowpass', fs=PlutoSamprate)     

        capture_range = updater.settings["chirp_count"]

        ticks = []
        transform = None
        debug = False

        new_col = None
        if not debug:
            try:
                pluto = SDR(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms, frame_length_samples)   
            except:
                updater.settings["debug"] = True
                updater.settings_changed = True
                updater.error_signal.emit(True)
                updater.error_signal.emit(False)
                return
            try:
                results = pluto.transmit_receive(iq=chirp, capture_range=capture_range, frame_length_samples=frame_length_samples)
            except:
                print("Error in transmit")
                updater.settings_changed = True
                return
            rx = np.sum(results, 0)

            tx = 100 * chirp
            beat_signal = tx * np.conj(rx)

            filtered = scipy.signal.filtfilt(b, a, beat_signal)

            filtered_fft = np.fft.fft(filtered, n=65536)#len(filtered)*zero_padding)
            new_col = filtered_fft / np.max(filtered_fft)
        else:
            y = np.linspace(0, 4 * np.pi, updater.buffer.shape[0])
            new_col = 0.5 + 0.5 * np.sin(y + time.time()) + 0.1 * np.random.randn(updater.buffer.shape[0]) + 0*1j
        try: 
            freq = np.linspace(0, PlutoSamprate,int(65536))
            C = 299792458
            distances = C * updater.settings["chirp_duration"] * freq / ( 2 * updater.settings["chirp_bandwidth"])
            updater.distances = distances
            updater.update_signal.emit((new_col, distances))

            print("New data was sent to plot")
        except RuntimeError:
                    pass
    updater.thread = Thread(target=run, daemon=True).start()


# --- Run ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    updater = ImageUpdater(app)
    updater.show()
    obtain_radar_data(updater)
    sys.exit(app.exec())
