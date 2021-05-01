import numpy as np
from matplotlib import ft2font
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.ndimage as spimg
from sumofsinesfitting import ipm_lin_solve
import scipy.fftpack as spfft
import neurokit2 as nk

BPM = 60
SEC, RATE = 2,500
n = SEC*RATE
CMP_RATE = 80
m = int(n*(100-CMP_RATE)/100)

#Simulating heart beat sensing
y = nk.ecg_simulate(duration=SEC, sampling_rate=RATE, heart_rate=BPM)

#Discrete cosine transform (DCT) to spectral domain
y_dct = spfft.dct(y, norm='ortho')

#Random sample ${CMP_RATE} points amongst n
cmp_sample = np.random.choice(n, m, replace=False)
cmp_sample.sort() 
y_cs = y[cmp_sample]

A = spfft.idct(np.identity(n), norm='ortho', axis=0)
A = A[cmp_sample]
print('Shape A',A.shape)
print('Rank A', np.linalg.matrix_rank(A.T))
print(y_cs)


x_cs = ipm_lin_solve(A, y_cs)
sig_ycs = spfft.idct(x_cs, norm='ortho', axis=0)

#Plotting

fig, axs = plt.subplots(2, 2)
fig.tight_layout(h_pad=2, w_pad=2)

axs[0,0].plot(y)
axs[0,0].set_xlabel('Time [ms]')
axs[0,0].set_ylabel('Amplitude')
axs[0,0].grid(True, which='both')

axs[0,1].plot(y_dct)
axs[0,1].set_xlabel('DCT')
axs[0,1].grid(True, which='both')

axs[1,0].plot(y)
axs[1,0].plot(sig_ycs)
axs[1,0].set_xlabel('Time [ms]')
axs[1,0].set_ylabel('Amplitude')

axs[1,1].imshow(A, cmap='hot', interpolation='nearest')
axs[1,1].set_title('A matrix')

axs[1,0].plot(sig_ycs[..., np.newaxis], color='#f2800e')
axs[1,0].set_xlabel('Time [ms]')
axs[1,0].set_ylabel('Amplitude')
axs[1,0].grid(True, which='both')

axs[1,1].plot(x_cs,  color='#f2800e')
axs[1,1].set_xlabel('DCT')
axs[1,1].grid(True, which='both')

print(np.linalg.norm(sig_ycs - y))
plt.show()

