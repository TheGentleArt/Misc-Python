# J.Williams
# In hopes of being used to filter steady-state(ish) data to filter out noise
# 'Denoising Data with FFT [Python]' from 'Steve Brunton' on YouTube was a big
#  help in making this.
import numpy as np; pi = np.pi
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename

# Open File --- uses a GUI window to ask user for file
root = tk.Tk()
root.withdraw()
filepath = askopenfilename()
root.destroy()  # Close the GUI window
df = pd.read_csv(filepath)  # Read data and store data in pandas dataframe

# Seperate data, pull sampling rate and find number of data points
df.columns = ['Time (s)', 'Amplitude (g)']  # Add column labels to dataframe
t = df.iloc[:, 0]  # Time (first column in csv file)
f = df.iloc[:, 1]  # Amplitude (second column in csv file)
n = len(t)  # Number of data points
dt = (t.iloc[-1] - t.iloc[0]) / n  # sample rate (time step) ; average change in time per sample, calculating instead of having user input sampling rate. using average since not sure if always going to be constant. if not constant though, not sure this script valid. need to look into this after finished writing
# f_delta = max(f) - min(f)  # Find the difference between max & min amplitude

# Find values of interest and print
accel_peak = max(abs(f))  # Max acceleration value seen
accel_rms = np.sqrt(np.mean(f**2))  # Root Mean Square acceleration value
accel_crest_factor = accel_peak / accel_rms  # Crest Factor (Ratio: peak/rms)
accel_std_dev = pd.DataFrame.std(f)  # Standard Deviation of acceleration
print("Peak accel :", accel_peak, "(g)")
print("RMS accel :", accel_rms, "(g)")
print("Crest factor :", accel_crest_factor)
print("Standard deviation :", accel_std_dev, "(g)")

# Get RMS data
# Create a window to iterate over which is effetively the sampling rate,
# Then find find out how many stepsit would take to get to get to end of time
# Then create a blank array and fill with a rolling rms ...type thing.
# Then store data in pandas series
window_width_rms = int(np.floor(1/dt))  # RMS calculation window width
steps_rms = int(np.floor(len(t)/window_width_rms))  # Number of steps for rms calculation
t_rms = np.zeros(steps_rms)  # Create blank array of time for rms calculation
f_rms = np.zeros(steps_rms)  # Create blank array of amplitude for rms calculation
for i in range(0, steps_rms):
    t_rms[i] = np.mean(t[(i*window_width_rms):((i+1)*window_width_rms)])
    f_rms[i] = np.sqrt(np.mean(f[(i*window_width_rms):((i+1)*window_width_rms)]**2))
t_rms = pd.Series(t_rms)  # Convert from numpy array to pandas series 
f_rms = pd.Series(f_rms)  # Convert from numpy array to pandas series 

# Compute the Fast Fourier Transform (FFT); break into components, add to df
fhat = np.fft.fft(f)  # Compute the fft
fhat = pd.Series(fhat)  # Convert from numpy array to pandas series
fhat_real = pd.DataFrame(np.real(fhat))  # Pulls the real component of fhat
fhat_imag = pd.DataFrame(np.imag(fhat))  # Pulls the imaginary component of fhat
df = df.assign(fhat_real=fhat_real)  # Add fft real numbers to dataframe, with label of fhat_real
df = df.assign(fhat_imag=fhat_imag)  # Add fft imaginary numbers to dataframe, with label of fhat_real
df = df.assign(fhat=fhat)  # Add fft result to dataframe, with label of fhat
# Do not need both components & complex fhat in df, but keeping for now ...

# Find Power Spectral Density; add to df
PSD = fhat * np.conj(fhat) / n  # Power Spectral Density. Units of amplitude^2/time
PSD = np.real(PSD)  # convert from complex to real number, since imag component should be zero
PSD = pd.Series(PSD)  # convert from numpy array to pandas series to easily add to dataframe
df = df.assign(PSD=PSD)  # Add power spectral density result to dataframe, with label of PSD

# Filter data using PSD values
PSDcutoff = 500  # Magnitude of PSD which we want to be the limit of high-pass filter
PSD_bool = PSD > PSDcutoff  # Creates a PSD boolean array, to be used for filtering data
fhat_clean = fhat * (PSD > PSDcutoff)  # Filter out fft data based on cutoff value of PSD

# Compute inverse FFT, to get cleaned data
ffilt = np.fft.ifft(fhat_clean)  # Inverse FFT for filtered time signal, to get cleaned data back
ffilt = np.real(ffilt)  # Convert from complex to real number, since imag component should be zero
ffilt = pd.Series(ffilt)  # Convert from numpy array to pandas series
df = df.assign(ffilt=ffilt)  #Add filtered data result to the dataframe

# Get RMS data of cleaned data
window_width_rms = int(np.floor(1/dt))  # RMS calculation window width
steps_rms = int(np.floor(len(t)/window_width_rms))  # Number of steps for rms calculation
t_rms_filt = np.zeros(steps_rms)  # Create blank array of time for rms calculation
f_rms_filt = np.zeros(steps_rms)  # Create blank array of amplitude for rms calculation
for i in range(0, steps_rms):
    t_rms_filt[i] = np.mean(t[(i*window_width_rms):((i+1)*window_width_rms)])
    f_rms_filt[i] = np.sqrt(np.mean(ffilt[(i*window_width_rms):((i+1)*window_width_rms)]**2))
t_rms_filt = pd.Series(t_rms_filt)  # Convert from numpy array to pandas series 
f_rms_filt = pd.Series(f_rms_filt)  # Convert from numpy array to pandas series

# Plot prep ... or something like that
freq = np.fft.fftfreq(n, dt)  # Create array of frequencies ; same as saying 'freq = (1 / (dt * n)) * np.arange(n)'
freq = pd.Series(freq)  # Convert frequency axis array to pandas series
freq = freq[np.arange(1, np.floor(n/2))]  # Shorten frequencies by half; limits axis of plot to the first half in order to not have mirror image of data on PSD plot
# #### want to look more into this to make sure this is right. would think these should have been integers usually
PSD = PSD[np.arange(1, np.floor(n/2))]  # Shorten PSD range by half to not have mirror image plotted later with negative values
# PSD_cutoff = pd.Series(np.repeat(PSDcutoff, n/2))
# freq_pass = freq * PSD_bool[np.arange(1, np.floor(n/2))] #Frequencies which passed the PSDcutoff filter


# =============================================================================
# ....need way to find frequency ranges that pass filter
# freq_pass = [i for i in range(len(freq)) if PSD.iloc[i] > PSDcutoff]  # list of frequencies that pass the PSDcutoff filter
# freq_pass = pd.Series(freq_pass)
# # #freq_high[0:int((len(freq_high)/2))]
# text_to_show = ['Min freq: ', freq_pass.min(), '; Max freq :', freq_pass.max()]
# =============================================================================


# Plot Setup
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(6, 1)  # Create a figure with (rows,cols)

# Create the first plot (Original data)
plt.sca(axs[0])  # First location of subplots
plt.plot(t, f, color='c', linewidth=1, label='Raw data')
plt.title("Original Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (g)")
plt0_xlim = np.floor(t.iloc[0]), t.iloc[-1]
plt.xlim(plt0_xlim)  # x limit of first plot, may want to look into changing
plt.legend()

# Create the second plot (RMS of original data)
plt.sca(axs[1])  # Second location of subplots
plt.plot(t_rms, f_rms, label='RMS')
plt.title("RMS of Original Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (g)")
plt0_xlim = np.floor(t.iloc[0]), t.iloc[-1]
plt.xlim(plt0_xlim)  # x limit of first plot, may want to look into changing
plt.legend()

# Create the third plot, the original data in the frequency domain after FFT
plt.sca(axs[2])  # Third location of the subplots
plt.plot(freq, PSD, color='c', linewidth=2, label='PSD')
plt.plot([freq.iloc[0], freq.iloc[-1]], [PSDcutoff, PSDcutoff], color='r', linewidth=1, label='PSDcutoff filter')
plt.xlim(np.floor(freq.iloc[0]), (np.ceil(freq.iloc[-1])))  # Change limits of plot to be just outside of bounds of freq
plt.title("Power Spectral Density (Original Data, in Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (g^2/Hz)")
plt.legend()

# =============================================================================
# Add label on chart for frequencies found
# text_loc_x = int(freq.iloc[-1]/3)  # x_location of text --- want to change this?
# text_loc_y = int(max(PSD)/2)  # y_location of text --- want to change this?
# plt.text(text_loc_x, text_loc_y, text_to_show)  # Add a text on plot stating the frequencies above the cutoff filter
# =============================================================================

# Create the fourth plot, spectrogram
plt.sca(axs[3])
plt.specgram(f, Fs=int(1/dt))
plt.title("Spectrogram (Original Data)")
plt.xlim(plt0_xlim)
plt.xlabel("Time (sec)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 2500)

# Create the fifth plot, with cleaned data in time domain
plt.sca(axs[4])  # Fourth location of the subplots
plt.plot(t, ffilt, color='k', linewidth=2, label='Cleaned')
plt.xlim(plt0_xlim)
plt.ylim(np.floor(np.min(ffilt)), np.ceil(max(ffilt)))
plt.title("Cleaned Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.legend()

# Create the sixth plot, with cleaned rms data in time domain
plt.sca(axs[5])  # Fourth location of the subplots
plt.plot(t_rms_filt, f_rms_filt, color='k', linewidth=2, label=' RMS Cleaned')
plt.xlim(plt0_xlim)
plt.ylim(0, np.ceil(max(f_rms_filt)))
plt.title("Cleaned Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.legend()

# Show the plot figure
plt.tight_layout()  # Change layout to make chart not run togeher as much
plt.show()  # Show the plots
