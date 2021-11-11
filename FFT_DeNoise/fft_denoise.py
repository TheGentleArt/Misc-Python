# J.Williams
# In hopes of being used to filter steady-state(ish) data to filter out noise
# Import data, plot data, show RMS of data, produce FFT of data, produce PSD of data, use a PSD cutoff to filter FFT data,
# then regenerate 'cleaned' data by means of inverse FFT, plot cleaned data, plot PSD and spectrogram.

# Main sources:
#   'Denoising Data with FFT [Python]','Steve Brunton', (YouTube)
#   https://blog.endaq.com/vibration-analysis-fft-psd-and-spectrogram

# To do:
#   Add multiple channels, maybe ask user for input on how many?
#   Dynamically change y-axis of some of the plots
#   Find out if there is a way of displaying the PSD plot, then asking user what value to set the filter at?
#   Look into other colormaps, default spectrogram colormap is hard to read to me
#   Change chart titles of cleaned data RMS chart.
#   Re-evaluate the cleaned data to make sure it makes sense. Seems to make the engine off data have a little more amplitude than originally, 
#   not sure if worth finding another way of editing based on near zero (non-event).
#   Re-evaluate RMS window size. Need to find out what is a proper window size here.
#   Perhaps give user option to filter data based on frequency range instead of PSD range?

# Import libraries
import numpy as np; pi = np.pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename

# Open File --- uses a GUI window to ask user for file
print("Select .csv file to open...")
print("If GUI for opening file is not seen, check if hidden behind program windows...")
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
dt = (t.iloc[-1] - t.iloc[0]) / n  # Sample rate (time step) ; average change in time per sample, calculating instead of having user input sampling rate. 
                                   # Using average since not sure if always going to be constant. if not constant though, not sure this script valid. 
                                   # Need to look into this after finished writing.
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
# Create a window to iterate over which is effectively the sampling rate,
# then find find out how many stepsit would take to get to get to end of time,
# then create a blank array and fill with a rolling rms ...type thing,
# then store data in pandas series
window_width_rms = int(np.floor(1/dt))  # RMS calculation window width (May need to look into this more)
steps_rms = int(np.floor(len(t)/window_width_rms))  # Number of steps for rms calculation
t_rms = np.zeros(steps_rms)  # Create blank array of time for rms calculation
f_rms = np.zeros(steps_rms)  # Create blank array of amplitude for rms calculation
for i in range(0, steps_rms):
    t_rms[i] = np.mean(t[(i*window_width_rms):((i+1)*window_width_rms)])
    f_rms[i] = np.sqrt(np.mean(f[(i*window_width_rms):((i+1)*window_width_rms)]**2))
t_rms = pd.Series(t_rms)  # Convert from numpy array to pandas series (may not need since this is not the same size as the other series in the df, but keeping for now)
f_rms = pd.Series(f_rms)  # Convert from numpy array to pandas series (may not need since this is not the same size as the other series in the df, but keeping for now)


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
PSD = fhat * np.conj(fhat) / n  # Power Spectral Density. Units of amplitude^2/time. ('Power' per frequency)
PSD = np.real(PSD)  # convert from complex to real number, since imag component should be zero
PSD = pd.Series(PSD)  # convert from numpy array to pandas series to easily add to dataframe
df = df.assign(PSD=PSD)  # Add power spectral density result to dataframe, with label of PSD

# Filter data using PSD values
PSDcutoff = 100  # Magnitude of PSD which we want to be the limit of high-pass filter
PSD_bool = PSD > PSDcutoff  # Creates a PSD boolean array, to be used for filtering data
fhat_clean = fhat * (PSD_bool)  # Filter out fft data based on cutoff value of PSD

# Compute inverse FFT, to get cleaned data
ffilt = np.fft.ifft(fhat_clean)  # Inverse FFT for filtered time signal, to get cleaned data back
ffilt = np.real(ffilt)  # Convert from complex to real number, since imag component should be zero
ffilt = pd.Series(ffilt)  # Convert from numpy array to pandas series
df = df.assign(ffilt=ffilt)  # Add filtered data result to the dataframe

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
fig1 = plt.figure(1, figsize=(12, 8))  # Create figure for plots
plt.rcParams.update({'font.size': 10})  # Dictate font size
gs = GridSpec(4, 2, figure=fig1)  # Create a grid to be used to place subplots

# Create the first plot (Original data)
ax1 = plt.subplot(gs[0, 0])
plt.plot(t, f, color='c', linewidth=1, label='Raw data')
plt.title("Original Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (g)")
plt0_xlim = np.floor(t.iloc[0]), t.iloc[-1]
plt.xlim(plt0_xlim)  # x limit of first plot, may want to look into changing
ax1.legend()

# Create the second plot (RMS of original data)
ax2 = plt.subplot(gs[1, 0])
plt.plot(t_rms, f_rms, label='RMS')
plt.title("RMS of Original Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (g)")
plt0_xlim = np.floor(t.iloc[0]), t.iloc[-1]
plt.xlim(plt0_xlim)  # x limit of first plot, may want to look into changing
ax2.legend()

# Create the third plot, the original data in the frequency domain after FFT
ax3 = plt.subplot(gs[2, 0])
plt.plot(freq, np.real(fhat[np.arange(1, np.floor(n/2))]), color='b')
plt.xlim(np.floor(freq.iloc[0]), (np.ceil(freq.iloc[-1])))  # Change limits of plot to be just outside of bounds of freq
plt.title("FFT (Original Data)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT")


# Create the fourth plot, the original data in the frequency domain after PSD
ax4 = plt.subplot(gs[3, 0])
plt.plot(freq, PSD, color='c', linewidth=2, label='PSD')
plt.plot([freq.iloc[0], freq.iloc[-1]], [PSDcutoff, PSDcutoff], color='r', linewidth=1, label='PSDcutoff filter')
plt.xlim(np.floor(freq.iloc[0]), (np.ceil(freq.iloc[-1])))  # Change limits of plot to be just outside of bounds of freq
plt.title("Power Spectral Density (Original Data, in Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (g^2/Hz)")
ax4.legend()

# =============================================================================
# Add label on chart for frequencies found
# text_loc_x = int(freq.iloc[-1]/3)  # x_location of text --- want to change this?
# text_loc_y = int(max(PSD)/2)  # y_location of text --- want to change this?
# plt.text(text_loc_x, text_loc_y, text_to_show)  # Add a text on plot stating the frequencies above the cutoff filter
# =============================================================================

# Create the fifth plot, with cleaned data in time domain
ax5 = plt.subplot(gs[0, 1])
plt.plot(t, ffilt, color='k', linewidth=2, label='Cleaned')
plt.xlim(plt0_xlim)
plt.ylim(np.floor(np.min(ffilt)), np.ceil(max(ffilt)))
plt.title("Cleaned Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (g)")
ax5.legend()

# Create the sixth plot, with cleaned rms data in time domain
ax6 = plt.subplot(gs[1, 1])
plt.plot(t_rms_filt, f_rms_filt, color='k', linewidth=2, label=' RMS Cleaned')
plt.xlim(plt0_xlim)
plt.ylim(0, np.ceil(max(f_rms_filt)))
plt.title("Cleaned Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (g)")
ax6.legend()

# Create the seventh plot, spectrogram
ax7 = plt.subplot(gs[2:, 1])  # Take up row 2 to the end, using col 1
plt.specgram(f, Fs=int(1/dt))
plt.title("Spectrogram (Original Data)")
plt.xlim(plt0_xlim)
plt.xlabel("Time (sec)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 2500)

# Show the plot figure
plt.tight_layout()  # Change layout to make chart not run together as much
plt.show()  # Show the plots
