"""
parse real data or matlab data
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal
import scipy.optimize
# TODO: scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

import denoise_algo


Fs = 500                        # sampling rate
SIGMA = 0.1                     # noise sigma
INITIAL_FIT_VALS = [600, 8]     # intial guess value for curve_fit
V_TH = 20                       # velocity threshold
DUR_TH = 0.024                  # duration threshold (s)
DUR_TH_N = int(DUR_TH * Fs)     # duration threshold (samples)
ADD_NOISE = True                # whether to add noise to the data


# set random seed
np.random.seed(0)


def main_squence(x_amp, eta, c):
    """
    formula for main sequence

    xdata: input
    V = eta*(1-e^(-A/c))
    """

    return eta * (1-np.exp(-x_amp/c))


def add_noise(signal, sigma):
    """
    add noise to signal
    """
    w = np.random.randn(*signal_data.shape)
    return signal + w * sigma


def diff(x):
    """
    central differenece filter
    Calculate derivative.
    """
    h = np.array([0.5, 0, -0.5])
    y = scipy.signal.convolve(x, Fs*h, mode='same')
    y[0] = 0
    y[-1] = 0
    return y


# load in data from .mat file
data = scipy.io.loadmat("simulation_type1.mat")
# print(data.keys())  # show what data is accessible
duration_data = data["DUR"][:, 0]
number_of_sacc = data["NSAC"][:, 0]
peak_velocity_data = data["PV"][:, 0]
reference_data = data["reference"][:, 0]
signal_data = data["signal"][:, 0]
time_data = data["t"][:, 0]
ampitude_data = data["AMP"][:, 0]
velocity_data = data["signal_vel"][:, 0]

# noisy data
if ADD_NOISE:
    noisy_p = add_noise(signal_data, SIGMA)
else:
    noisy_p = signal_data

# get velocity from position
noisy_v = diff(noisy_p)

exit()
# TODO
# get array of 1 and 0's indicating saccades
# calculate sigma using array
# calculate alpha and beta
# denoise signal
denoised_signal = denoise_algo.cgtv()


# create figure
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.set_ylabel("Position (deg)")
ax0.set_xlabel("Time (s)")
# TODO below: change 10 to number of saccades
ax0.set_title('CGTV: %d saccades detected' % 10)

# plot original signal
ax0.plot(time_data, signal_data, 'k-', label='original signal')
ax0.plot(duration_data, 'r.')

plt.plot()
plt.legend()
plt.show()
