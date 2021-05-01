import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt

PI2 = 2*np.pi

def square_wave(f,t):
    if hasattr(t, "__len__"):
        return [np.sign(np.sin(PI2*f*i)) for i in t]
    return np.sign(np.sin(PI2*f*t))

def triangle_wave(f,t):
    if hasattr(t, "__len__"):
        return [2/np.pi * np.arcsin(np.sin(PI2*f*i)) for i in t]
    return 2/np.pi * np.arcsin(np.sin(PI2*f*t))

def sawtooth_wave(f,t):
    if hasattr(t, "__len__"):
        return [-2/np.pi * np.arctan(np.tan(i*np.pi*f)**-1) for i in t]
    return -2/np.pi * np.arctan(np.tan(t*np.pi*f)**-1)
