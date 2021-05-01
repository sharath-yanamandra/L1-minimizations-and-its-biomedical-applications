import numpy as np
from matplotlib import ft2font
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.ndimage as spimg
import scipy.misc as smisc
from sumofsinesfitting import ipm_lin_solve
import scipy.fftpack as spfft
import neurokit2 as nk
from cv2 import cv2

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed
MRI_PATH = r'C:\Users\Utente\Desktop\AI Unipi\I I\Computational Mathematics\CompressedSensing\scan.png'

im_gray = cv2.imread(MRI_PATH, cv2.IMREAD_GRAYSCALE)
X = spimg.zoom(im_gray, 0.05)
ny,nx = X.shape

# extract small sample of signal
k = round(nx * ny * 0.5) # 50% sample
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
b = X.T.flat[ri]
b = np.expand_dims(b, axis=1)

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
A = A[ri,:] # same as phi times kron

Xat2 = ipm_lin_solve(A, b)

Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)

# confirm solution
if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    print('Warning: values at sample indices don\'t match original.')

# create images of mask (for visualization)
mask = np.zeros(X.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]

cv2.imshow('ImageWindow', Xat)
cv2.waitKey()
