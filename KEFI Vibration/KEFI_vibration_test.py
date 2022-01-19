# J.Williams

# Import libraries
import numpy as np; pi = np.pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename
import os



# Open File --- uses a GUI window to ask user for file
print("\nSelect .csv file to open...")
print("If GUI for opening file is not seen, check if hidden behind program windows...\n")
root = tk.Tk()
root.withdraw()
filepath_open = askopenfilename()
root.destroy()  # Close the GUI window
print("Opening: " + filepath_open)

# File naming etc.
dir_path_original_file = os.path.dirname(filepath_open)
#This requires the folder for the text files that are being analysed to have these keywords in them...
# Determine testing type
if "NEUTRAL" in filepath_open.upper():
    testing_config = "Neutral"
elif "ACCELERATION" in filepath_open.upper():
    testing_config = "Acceleration"
elif "ENGINE SPEED" in filepath_open.upper():
    testing_config = "Engine Speed"
# Determine vehicle configuration
if "AXLE" in dir_path_original_file.upper():
    vehicle_config = "Stock Flywheel with Axle Pads"
elif "STOCK" in dir_path_original_file.upper():
    vehicle_config = 'Stock Flywheel'
elif "LIGHT" in dir_path_original_file.upper():
    vehicle_config = "Lightweight Flywheel"

# Create folders for filepaths if not already there
filePath_testing_config = 'U:\\Departments\\Engineering\\Users\\Justin Williams\\Python\\Kawi Vibration Test\\Exported Files\\' + testing_config
filePath_vehicle_config = 'U:\\Departments\\Engineering\\Users\\Justin Williams\\Python\\Kawi Vibration Test\\Exported Files\\' + testing_config + '\\' + vehicle_config
# Create testing type directory if not already there
if not os.path.exists(filePath_testing_config):
    os.makedirs(filePath_testing_config)
# Create vehicle configuration type directory if not already there
if not os.path.exists(filePath_vehicle_config):
    os.makedirs(filePath_vehicle_config)


# Bring data into a dataframe
print("\nSetting up dataframe...")
df = pd.read_csv(filepath_open, sep='\t')  # Read data and store data in pandas dataframe
print("Raw data setup in dataframe. \n")

# Re-label channels
print("Re-labelling channels for clarity...")
df.columns = ['Time (s)',
              'Floorboard (x-dir)', 'Floorboard (y-dir)', 'Floorboard (z-dir)',
              'Steering Wheel Right (x-dir)', 'Steering Wheel Right (y-dir)',
              'Steering Wheel Right (z-dir)',
              'Steering Wheel Left (x-dir)', 'Steering Wheel Left (y-dir)',
              'Steering Wheel Left (z-dir)',
              'Seat Bottom (x-dir)', 'Seat Bottom (y-dir)',
              'Seat Bottom (z-dir)',
              'Seat Back (x-dir)', 'Seat Back (y-dir)', 'Seat Back (z-dir)',
              'Cyl Head (x-dir)', 'Cyl Head (y-dir)', 'Cyl Head (z-dir)',
              'Engine Pan (x-dir)', 'Engine Pan (y-dir)', 'Engine Pan (z-dir)',
              'Wheel Speed (Rear Passenger) (rpm)', 'Wheel Speed (Rear Driver) (rpm)',
              'Engine Speed (rpm)', 'Latitude', 'Longitude',
              'GPS Speed (mph)', 'GPS Satellite Count']  # Add column labels to dataframe
print("Channels relabeled. \n")

# Save re-labeled original data in U drive folder
# =============================================================================
#  File path levels are as follows:
#     Exported (1)
#         Testing Configuration (2)
#             Vehicle Configuration (3)
#                 Plots (4)
#                 Resultant Calcs (4)
#
# For example, filepath_4 = Exported\Testing config\veh config\, etc.
# =============================================================================

print("Saving relabelled original dataframe data to csv in path: \n'" + filePath_vehicle_config + "' \nThis may take awhile...")
df.to_csv(filePath_vehicle_config+'\\Original_Data_Relabelled.csv') #####
print("Save successful. \n")

# Calculate resultants, drop other columns
print("Calculating resultants and adding to dataframe...")
df['Avg Wheel Speed'] = (df['Wheel Speed (Rear Driver) (rpm)'] +
                         df['Wheel Speed (Rear Passenger) (rpm)']) / 2

df['Engine Speed (Hz)'] = df['Engine Speed (rpm)']/60

# Calculate Resultants of 3-dims to 1-dim
df['Floorboard Resultant (g)'] = (df['Floorboard (x-dir)']**2 +
                              df['Floorboard (y-dir)']**2 +
                              df['Floorboard (z-dir)']**2)**(1/2)

df['Steering Wheel Right Resultant (g)'] = (df['Steering Wheel Right (x-dir)']**2 +
                                        df['Steering Wheel Right (y-dir)']**2 +
                                        df['Steering Wheel Right (z-dir)']**2)\
                                        ** (1/2)

df['Steering Wheel Left Resultant (g)'] = (df['Steering Wheel Left (x-dir)']**2 +
                                       df['Steering Wheel Left (y-dir)']**2 +
                                       df['Steering Wheel Left (z-dir)']**2)\
                                       ** (1/2)

df['Seat Bottom Resultant (g)'] = (df['Seat Bottom (x-dir)']**2 +
                               df['Seat Bottom (y-dir)']**2 +
                               df['Seat Bottom (z-dir)']**2)**(1/2)

df['Seat Back Resultant (g)'] = (df['Seat Back (x-dir)']**2 +
                             df['Seat Back (y-dir)']**2 +
                             df['Seat Back (z-dir)']**2)**(1/2)

df['Seat Back Resultant (g)'] = (df['Seat Back (x-dir)']**2 +
                             df['Seat Back (y-dir)']**2 +
                             df['Seat Back (z-dir)']**2)**(1/2)

df['Cyl Head Resultant (g)'] = (df['Cyl Head (x-dir)']**2 +
                            df['Cyl Head (y-dir)']**2 +
                            df['Cyl Head (z-dir)']**2)**(1/2)

df['Engine Pan Resultant (g)'] = (df['Engine Pan (x-dir)']**2 +
                              df['Engine Pan (y-dir)']**2 +
                              df['Engine Pan (z-dir)']**2)**(1/2)
print("Resultants computed.\n")

# Drop df columns that are single axis accel data, etc.
print("Dropping original data from dataframe, and re-ordering channels...")
df = df.drop(df.columns[1:24], axis = 1)
df = df.drop(df.columns[2:4], axis = 1)
df = df.drop(df.columns[3], axis = 1)
df_cols = df.columns[[0,1,4,10,11,5,8,9,6,7,3,2]] # Create list to re-order cols
df = df[df_cols]
print("Dataframe size reduced. Columns re-ordered. \n")


# Save resultants to file
print("Saving resultants dataframe data to csv in path: \n'" + filePath_vehicle_config + "'\nThis may take awhile...")
df.to_csv(filePath_vehicle_config + '\\Calculated_Resultant_Data.csv') #save resultants to a file #####
print("Save successful. \n")

# May want to do math to make dynamic, but worry if data is not uniform...
sample_rate = 5000  # Hz
delta_t = 1/5000 # seconds
print("Sample rate is hardcoded as: "+ str(sample_rate) + " Hz. \n")


# Filter out data not within engine speed range
filt_low_rpm = 1  # rpm
filt_high_rpm = 3250  # rpm
median_eng_freq = (filt_low_rpm + filt_high_rpm) / (2*60)
print("Filtering out data when engine speed is not between " +
      str(filt_low_rpm) + " and " + str(filt_high_rpm) + " rpm...")
filt_eng_spd = (df['Engine Speed (rpm)'] >= filt_low_rpm) & (df['Engine Speed (rpm)'] <= filt_high_rpm)  # Boolean array
df = df[filt_eng_spd]  # df is now only rows where above boolean is true
t_new = df.shape[0] / sample_rate
df.rename(columns={'Time (s)':'Original Time (s)'}, inplace=True)
df.insert(0,'New Time (s)',np.arange(0,t_new,delta_t))
print("Data filtered.\n")
print("Saving filtered data to csv in path: \n'" + filePath_vehicle_config + "'\nThis may take awhile...")
df.to_csv(filePath_vehicle_config + '\\Filtered_Resultant_Data_JBW.csv') # Save filtered data to a file #####
print("Filtered data saved. \n")

# Find average filtered engine speed for use in plots later
avg_filt_eng_spd = np.average(df['Engine Speed (rpm)'])
avg_filt_eng_spd_freq = avg_filt_eng_spd/60

# Loop through each channel
itr = 4 #starting column index for loop (since first few columns (like time) do not need to be looped through)
t = df.iloc[:,0]  # Set t as 'new time' (after filtered engine speed), does not need to iterate
for i in df.columns:
    if itr < df.shape[1]: # just using this logic check to not get error after looping to end
        df_itr = df.iloc[:,[0,2,3,itr]] #New df with new time, engine speed (in rpm and Hz), and channel to be analyzed
        if "Resultant" in df_itr.columns[-1]: # Only do the below if resultant is in the name of the channel being looped through. Doing this to not analyze wheel speed etc with FFTs
            f = df_itr.iloc[:,3]  # set f as the accelerometer data column
            name_prefix = df.columns[itr] #for filename later
            print("Looping through ", name_prefix, " ... \n")
            n = len(t)  # Number of data points
            dt = (t.iloc[-1] - t.iloc[0]) / (n-1)  # Sample rate (time step) ; average change in time per sample, calculating instead of having user input sampling rate.
                                               # Using average since not sure if always going to be constant. if not constant though, not sure this script valid.
                                               # Need to look into this after finished writing.
            # f_delta = max(f) - min(f)  # Find the difference between max & min amplitude

            # Find values of interest and print
            print("   Computing peak, RMS, crest factor, and standard deviation...")
            accel_peak = max(abs(f))  # Max acceleration value seen
            accel_rms = np.sqrt(np.mean(f**2))  # Root Mean Square acceleration value
            accel_crest_factor = accel_peak / accel_rms  # Crest Factor (Ratio: peak/rms)
            accel_std_dev = pd.DataFrame.std(f)  # Standard Deviation of acceleration
            print("     Peak accel :", accel_peak, "(g)")
            print("     RMS accel :", accel_rms, "(g)")
            print("     Crest factor :", accel_crest_factor)
            print("     Standard deviation :", accel_std_dev, "(g) \n")

            # Get RMS data
            # Create a window to iterate over which is effectively the sampling rate,
            # then find find out how many stepsit would take to get to get to end of time,
            # then create a blank array and fill with a rolling rms ...type thing,
            # then store data in pandas series
            print("   Computing RMS data...")
            ###below last value need to make variable (window width control)
            window_width_rms = int(np.floor(1/dt)/10)  # RMS calculation window width (May need to look into this more)
            steps_rms = int(np.floor(len(t)/window_width_rms))  # Number of steps for rms calculation
            t_rms = np.zeros(steps_rms)  # Create blank array of time for rms calculation
            f_rms = np.zeros(steps_rms)  # Create blank array of amplitude for rms calculation
            for i in range(0, steps_rms):
                t_rms[i] = np.mean(t[(i*window_width_rms):((i+1)*window_width_rms)])
                f_rms[i] = np.sqrt(np.mean(f[(i*window_width_rms):((i+1)*window_width_rms)]**2))
            t_rms = pd.Series(t_rms)  # Convert from numpy array to pandas series (may not need since this is not the same size as the other series in the df, but keeping for now)
            f_rms = pd.Series(f_rms)  # Convert from numpy array to pandas series (may not need since this is not the same size as the other series in the df, but keeping for now)
            print("   RMS computed. \n")

            # Add frequency column for excel file


            # Compute the Fast Fourier Transform (FFT); break into components, add to df
            print("   Computing FFT...")
            fhat = np.fft.fft(f)  # Compute the fft
            fhat = pd.Series(fhat)  # Convert from numpy array to pandas series
            fhat_real = pd.DataFrame(np.real(fhat))  # Pulls the real component of fhat
            fhat_imag = pd.DataFrame(np.imag(fhat))  # Pulls the imaginary component of fhat
            df_itr = df_itr.assign(fhat_real=fhat_real)  # Add fft real numbers to dataframe, with label of fhat_real
            df_itr = df_itr.assign(fhat_imag=fhat_imag)  # Add fft imaginary numbers to dataframe, with label of fhat_real
            df_itr = df_itr.assign(fhat=fhat)  # Add fft result to dataframe, with label of fhat
            print("   FFT computed.\n")
            # Do not need both components & complex fhat in df, but keeping for now ...

            # Find Power Spectral Density; add to df
            print("   Computing PSD...")
            PSD = fhat * np.conj(fhat) / n  # Power Spectral Density. Units of amplitude^2/time. ('Power' per frequency)
            PSD = np.real(PSD)  # convert from complex to real number, since imag component should be zero
            PSD = pd.Series(PSD)  # convert from numpy array to pandas series to easily add to dataframe
            df_itr = df_itr.assign(PSD=PSD)  # Add power spectral density result to dataframe, with label of PSD
            print("   PSD computed.\n")

            #df = df.iloc[:,[]]

            # Save individual accelerometer calculations to file
            outputFileName = str(name_prefix + " Calcs")
            resultantSubFolderName = "Resultant Channel Calcs"
            if not os.path.exists(filePath_vehicle_config + '\\' + resultantSubFolderName + '\\'): # Check if filepath is there, if not create
                os.makedirs(filePath_vehicle_config + '\\' + resultantSubFolderName)
            df_itr.to_csv(filePath_vehicle_config + '\\' + resultantSubFolderName + '\\' + outputFileName + ".csv", sep=',')
            print("   " + outputFileName + " saved. \n")

            ### this area below is new
            ### this area below is new
            ### this area below is new
            ### this area below is new
            ### this area below is new

            #Plot setup
            print("   Creating frequency array ...")
            freq = np.fft.fftfreq(n, dt)  # Create array of frequencies ; same as saying 'freq = (1 / (dt * n)) * np.arange(n)'
            freq = pd.Series(freq)  # Convert frequency axis array to pandas series
            freq = freq[np.arange(1, np.floor(n/2))]  # Shorten frequencies by half; limits axis of plot to the first half in order to not have mirror image of data on PSD plot
            print("   Frequency array created. \n")
            print("   Setting up plot ...")
            fig1 = plt.figure(1, figsize=(12, 8))  # Create figure for plots
            plt.rcParams.update({'font.size': 10})  # Dictate font size
            gs = GridSpec(4, 2, figure=fig1)  # Create a grid to be used to place subplots

            # Define x limits of plots
            orders_to_plot = 4
            x_lim_t = np.floor(t.iloc[0]), t.iloc[-1]
            x_lim_freq = (0,median_eng_freq*orders_to_plot)

            # x-axis tick marks at half orders
            major_ticks = np.arange(0, np.ceil(x_lim_freq[1]), avg_filt_eng_spd_freq)
            minor_ticks = np.arange(0, np.ceil(x_lim_freq[1]), avg_filt_eng_spd_freq/2)


            # Create first plot (RMS of data vs time)
            ax1 = plt.subplot(gs[0, 0])
            plt.plot(t_rms, f_rms, color='c', linewidth=1, label='RMS')
            plt.title("RMS of filtered data vs time")
            plt.xlabel("Time (sec)")
            plt.ylabel("Amplitude (g)")
            plt0_xlim = np.floor(t.iloc[0]), t.iloc[-1]
            plt.xlim(plt0_xlim)  # x limit of first plot, may want to look into changing
            ax1.legend()

            # Create the second plot (Engine Speed vs time)
            ax2 = plt.subplot(gs[1, 0])
            plt.plot(t, df_itr['Engine Speed (rpm)'], linewidth=1)
            plt.title("Engine speed vs time")
            plt.xlabel("Time (sec)")
            plt.ylabel("rpm")
            plt0_xlim = np.floor(t.iloc[0]), t.iloc[-1]
            plt.xlim(x_lim_t)  # x limit of first plot, may want to look into changing
            plt.ylim(filt_low_rpm, filt_high_rpm)
            plt.hlines(avg_filt_eng_spd, x_lim_t[0], x_lim_t[1], linestyles='dashed', label = 'AvG Eng Spd') #Create horizontal line for avg engine speed
            ax2.legend()
            # Create additional y axis for second plot (Engine speed (Hz) vs time)
            ax2_2 = ax2.twinx()
            ax2_2.set_ylim(filt_low_rpm/60,filt_high_rpm/60)
            ax2_2.set_ylabel('Hz')
            ax2_2.set_yticks(np.linspace(filt_low_rpm/60, filt_high_rpm/60, 4))

            # Create third plot (FFT)
            ax3 = plt.subplot(gs[2, 0])
            plt.plot(freq, np.real(fhat[np.arange(1, np.floor(n/2))]), color='b')
            #plt.xlim(np.floor(freq.iloc[0]), (np.ceil(freq.iloc[-1])))  # Change limits of plot to be just outside of bounds of freq
            plt.xlim(x_lim_freq)
            plt.ylim(0,2000)
            plt.title("FFT")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (Need to fix scaling issue)")
            ax3.set_xticks(major_ticks)
            ax3.set_xticks(minor_ticks, minor=True)

            # Create the fourth plot, PSD
            ax4 = plt.subplot(gs[3, 0])
            plt.plot(freq, np.real(PSD[np.arange(1, np.floor(n/2))]), color='c', linewidth=1.5, label='PSD')
            #plt.xlim(np.floor(freq.iloc[0]), (np.ceil(freq.iloc[-1])))  # Change limits of plot to be just outside of bounds of freq
            #plt.hlines(np.percentile(PSD, 50), x_lim_freq[0], x_lim_freq[1], color='k', label = 'Median PSD') # debating cutting data below this line out
            plt.xlim(x_lim_freq)
            plt.ylim(0,25)
            plt.title("Power Spectral Density (Original Data, in Frequency Domain)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (g^2/Hz)")
            ax4.set_xticks(major_ticks)
            ax4.set_xticks(minor_ticks, minor=True)
            #plt.xticks(np.arange(0,200,19))

            # Create the fifth plot, spectrogram
            ax5 = plt.subplot(gs[0:, 1])  # Take up row 0 to the end, using col 1
            plt.specgram(f, Fs=int(1/dt), cmap='viridis')
            plt.plot(t, df_itr['Engine Speed (rpm)']/60, linewidth=0.2, label='Engine Speed 1st Order')
            plt.plot(t, df_itr['Engine Speed (rpm)']/60*2, linewidth=0.15, label='Engine Speed 2nd Order')
            plt.plot(t, df_itr['Engine Speed (rpm)']/60*3, linewidth=0.1, label='Engine Speed 3rd Order')
            plt.title("Spectrogram (Filtered Data)")
            plt.xlim(x_lim_t)
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim(0, median_eng_freq*orders_to_plot*1.5) # Goes a bit beyond orders given
            ax5.legend()

            # Show the plot figure
            plt.tight_layout()  # Change layout to make chart not run together as much
            if not os.path.exists(filePath_vehicle_config + '\\Plots'): # Check if filepath is there, if not create
                os.makedirs(filePath_vehicle_config + '\\Plots')
            plt.savefig(filePath_vehicle_config + '\\Plots\\' + df_itr.columns[3] + ' Plot.png')
            plt.show()  # Show the plots

            print("   Plot completed and saved. \n")

    itr = itr + 1

print("\nLooping finished.\nDone.")
