import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt
import waves as fun

I = [0,1]
rate, f = 1000, 4

t = np.linspace(*I, rate, endpoint = True)
a = fun.sawtooth_wave(f,t)

plt.figure()
plt.title('{} Hz sampling {} Hz/s'.format(f, rate))
plt.plot(t, a, label="Sawtooth Wave", color='#0d57d6')
plt.legend(loc="upper left")
plt.ylim(-2, 2)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True, which='both', alpha = 0.6)
plt.show()