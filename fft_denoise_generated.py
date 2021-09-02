#'Denoising Data with FFT [Python]' from 'Steve Brunton' on YouTube was used to make this

import numpy as np ; pi = np.pi
import matplotlib.pyplot as plt
from tkinter import filedialog


#Create a simple signal with two frequencies
fs = 1000 #sampling frequency (Hz)
dt = 1/fs #time per samples
t = np.arange(0,1,dt) #create time vector
f=np.sin(2*pi*50*t) + np.sin(2*pi*120*t) + np.cos(2*pi*3.8*t) #sum of three frequencies
f_clean = f #save the original signal generated as a clean version
f = f + 2.5*np.random.randn(len(t)) # add noise to signal generated

#Compute the Fast Fourier Transform (FFT)
n = len(t) #number of data points
fhat = np.fft.fft(f, n) #(f,n) #compute the fft ; input data with frequency (f) and number of points (n) ; outputs fhat (complex value fourier coefficients (magnitude and phase))
#############
print("size of fhat:", fhat.shape)
#############
PSD = fhat * np.conj(fhat) / n #Power Spectrum Density (power per frequency)
freq = (1/(dt*n)) * np.arange(n) #create x-axis of frequencies
L = np.arange(1,np.floor(n/2),dtype='int') #Only plot the first half of

#Filter Data using PSD
#PSDcutoff value will need to be updated, edit after seeing plots
PSDcutoff = 60 #amplitude of PSD which we want to be cutoff for filter
indices = PSD > PSDcutoff #Find all frequencies with large Power
#PSDclean = PSD * indices #(Did not use so far) Zero out all other frequencies
fhat = indices * fhat #Zero out small Fourier coefficients in Y
ffilt = np.fft.ifft(fhat) #Inverse FFT for filtered time signal, to get cleaned data back

#Find the discrete values of the frequencies which passed the PSD cutoff filter
freq_high = [i for i in range(len(freq)) if PSD[i] > PSDcutoff]
freq_high = freq_high[0:int((len(freq_high)/2))]

#Plot prep --- Convert from complex number to real number to avoid plot warning
PSD = np.real(PSD)
ffilt = np.real(ffilt)

#Plot
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 10})
fig,axs = plt.subplots(3,1) #Create a figure with (rows,cols)
#create the first plot, the original data in time domain
plt.sca(axs[0]) #first location
plt.plot(t,f,color='c',linewidth=1.5,label='Noisy generated data')
plt.plot(t,f_clean,color='k',linewidth=2,label='Clean generated data before injecting noise')
plt.xlim(t[0],t[-1])
plt.ylim(-5,5) #######
plt.title("Original Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.legend()
#create the second plot, the original data in the frequency domain after FFT
plt.sca(axs[1]) #second location
plt.plot(freq[L],PSD[L],color='c',linewidth=2,label='Noisy')
plt.plot([L[0],L[-1]],[PSDcutoff,PSDcutoff],color='r',linewidth=2,label='PSDcutoff filter')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.title("Original Data, in Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("'Power'")
plt.legend()
#add label on chart for frequencies found
text_loc_x = int(len(freq)/4) #x_location of text --- want to change this
plt.text(text_loc_x,max(PSD)/2,freq_high) #Add a text on plot stating the frequencies above the cutoff filter
#move onto third plot
#Create the third plot, with cleaned data in time domain
plt.sca(axs[2]) #third location
plt.plot(t,ffilt,color='k',linewidth=2,label='Cleaned')
plt.xlim(t[0],t[-1])
plt.ylim(-5,5) ############
plt.title("Cleaned Data")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.legend()
#show the plots
plt.tight_layout()
plt.show()
