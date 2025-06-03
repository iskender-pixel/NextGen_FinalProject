clear
clc

t_start = tic;
save_path = fileparts(mfilename("fullpath"));

%%%%%%%%%%%%%%%%%%%% Python Environment configuration %%%%%%%%%%%%%%%%%%%%%

terminate(pyenv);           % Terminate any active Python sessions
pe = pyenv;                 % Initialize Python environment
if pe.Status == "NotLoaded"
    disp("----- Python Environment Configuration -----");
    pyenv(ExecutionMode="OutOfProcess", Version="3.12");
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Pluto's parameters configuration %%%%%%%%%%%%%%%%%%%%%

% If you want repeatable alignment between transmit and receive then the 
% rx_lo, tx_lo and sample_rate can only be set once after power up. If you
% want to change the following parameters' values after the first run you 
% must reboot the Pluto (disconnect and reconnect it)

Pluto_IP = '192.168.2.1';
PlutoSamprate = 60.5e6;  % Sampling frequency (Hz): Pluto can sample up to
%                       61 MHz but due to the USB 2.0 interface you have to
%                       choose values lower or equal to 5MHz if you want to
%                       receive 100% of samples over time.
centerFrequency = 2.5e9;  % Pluto operating frequency (Hz) must be between
%                         70MHz and 6GHz
tx_gain = -20; %-45  % Pluto TX channel Gain must be between 0 and -88
rx_gain = 40;% 0   % Pluto RX channel Gain must be between -3 and 70

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% TX Waveform generation %%%%%%%%%%%%%%%%%%%%%%%%%%
B = 30e6;       % Chirp bandwidth (Hz)
T = 100e-6;       % Chirp duration (s)
f0 = 2.5e9;      % Carrier frequency of the RF signal (Hz)
fs = PlutoSamprate;     % Sampling rate of all the signals in this simulation (Hz) 
c = 3e8;        % Speed of light (m/s)


chirp_duration = 1; % Chirp duration (ms)
% t = 1/fs:1/fs:chirp_duration*1e-3;  
t = single(0:1/fs:(chirp_duration*1e-3 ));  

% t = 0:1/fs:(T-1/fs); 
k = B / T;     % Chirp slope defined as ratio of bandwidth over duration
sig_A = exp(1j*2*pi*(0.5*k*t.^2));

tx_waveform = 2^12.*sig_A.';  % # The PlutoSDR expects samples to be between 
%                           -2^14 and +2^14, not -1 and +1 like some SDRs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% SDR Object Initialization %%%%%%%%%%%%%%%%%%%%%%%%%

% The following variable will be passed to the setupPlutoSDR_TDD python
% module to determine the size of the RX buffer and to evaluate how many
% samples must be received in one frame
rx_time_ms = 1000*length(t)/PlutoSamprate; 

% Setup SDR with TDD engine
sdr_object = py.setupPlutoSDR_TDD.initialize_Pluto_TDD(Pluto_IP, PlutoSamprate, centerFrequency, rx_gain, tx_gain, rx_time_ms);

% SDR and TDD objects; these object must be passed to the Python module 
% that will be used to transmit and receive data.
my_sdr = sdr_object{1};
tddn = sdr_object{2};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Signals trasmission and reception %%%%%%%%%%%%%%%%%%%%

% The following variables will be passed to the TDD_Transreceiver python
% module to determine how many samples must be received in one frame and
% how many frames must be acquired
frame_length_samples = floor(rx_time_ms*PlutoSamprate/1000);

capture_range = 100;

% Send and receive data with Pluto
results = py.TDD_Transreceiver.pluto_transmit_receive(my_sdr, tddn, tx_waveform, int32(capture_range), int32(frame_length_samples),save_path);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the received data into the current Matlab workspace
received_data=load([save_path,'\received_data.mat']);
received_data=received_data.received_data;

t_stop = toc(t_start);

disp([num2str(capture_range), ' Frames received - Elapsed Time: ', num2str(t_stop),'s']);


%%

T = 1/fs; %time interval between sampling points
L= length(received_data); %Length of the waveform received
t = (0:L-1)*T; % Creating of the Time vector
tx_waveform(end)=[];


s_beat = received_data .* conj(tx_waveform.'); % Mixing 
s_beat = lowpass(real(s_beat), B/2, fs); % Low-pass filter
sig_C=s_beat;
%%
sig_C_fft=fftshift(fft(sig_C(1,:)));
freqaxis=linspace(0,fs,length(sig_C_fft)); %in Hz
figure;
plot(freqaxis.*1e-6,20*log10(abs(sig_C_fft)./max(abs(sig_C_fft)))); 
title('Signal spectrum of first chirp'); xlabel('Frequency (MHz)');
% xlim([0 5])


sug_C_fft_sum=(fftshift(fft(sum(sig_C,1),[],2),2));
freqaxis=linspace(0,fs,length(sug_C_fft_sum)); %in Hz
figure;
plot(freqaxis.*1e-6,20*log10(abs(sug_C_fft_sum)./max(abs(sug_C_fft_sum)))); 
title('Signal spectrum with 100 chirps'); xlabel('Frequency (MHz)');
% xlim([0 5])
