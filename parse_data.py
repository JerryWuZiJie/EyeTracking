"""
parse real data or matlab data
TODO: np.nonzero https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
TODO: scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
TODO: indexing: np.ix_(): https://stackoverflow.com/a/11393946/13720936
TODO: check saccade_cgtv.m for reference
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.optimize
import scipy.stats

import algorithms

#
Fs = 500                        # sampling rate
INITIAL_FIT_VALS = [600, 8]     # intial guess value for curve_fit
ITER_N = 20                     # number of iteration times

# parameter used for simulation data
SIMULATION = True               # whether it's a simulation)
SIGMA = 0.1                     # noise sigma for test data (TODO)

# parameter used for estimation of sigma, amplitude, and duration
MOVAVG_FS = 10
V_TH = 10                       # velocity threshold
DUR_TH = 0.024                  # duration threshold (s)
FIX_TH = 0.04                   # fixation threshold (s)

# set random seed
np.random.seed(0)


def add_noise(signal, sigma):
    """
    add noise to signal
    """
    w = np.random.randn(*signal_data.shape)
    return signal + w * sigma


### load in data from .mat file ###
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
if SIMULATION:
    noisy_p = add_noise(signal_data, SIGMA)
else:
    noisy_p = signal_data

### calculate coefficient alpha and beta ###
# run VT algorithm to get detection array, 1 for saccade and 0 for fixation
est_detect_array, est_total_sac = algorithms.VT(
    noisy_p, Fs, V_TH, DUR_TH, FIX_TH)
# get standard deviation for all fixations, this will be the signma
# subtract average for each fixation
fix_len = 0  # len of each fixation
fixations = np.copy(noisy_p)
for i in range(len(est_detect_array)):
    if est_detect_array[i] == 0:
        fix_len += 1
    elif fix_len != 0:
        fixations[i-fix_len:i] -= np.mean(noisy_p[i-fix_len:i])
        fix_len = 0
est_sigma = scipy.stats.tstd(fixations[est_detect_array == 0])
# get average duration
est_total_dur = (est_detect_array == 1).sum()
est_dur = est_total_dur / est_total_sac / Fs
# get average amplitude TODO: seems wrong, check saccade_cgtv 83 amp_avg?
est_v_smooth = algorithms.v_denoise(noisy_p, Fs)
est_total_amp = np.abs(est_v_smooth[est_detect_array == 1]).sum()
est_amp = est_total_amp / est_total_sac / Fs
# calculate sigma using equation from the paper
if Fs <= 500:
    alpha = 0.016 * Fs * est_sigma
    beta = 0.008 * Fs * np.sqrt(est_amp) * np.exp(5 * est_dur)
else:
    alpha = (0.0032*Fs + 6.4) * est_sigma
    beta = (0.0016*Fs + 3.2) * np.sqrt(est_amp) * np.exp(5 * est_dur)

# denoise signal
denoised_signal = algorithms.cgtv(noisy_p, alpha, beta, ITER_N)

# run VT algorithm on smooth out signal
detection_array, total_sacs = algorithms.VT(
    denoised_signal, Fs, V_TH, DUR_TH, FIX_TH)

# create figure
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.set_ylabel("Position (deg)")
ax0.set_xlabel("Time (s)")
ax0.set_title('CGTV: %d saccades detected' % total_sacs)

# plot detection
ax0.plot(time_data, detection_array*10, label="detection")

# plot original signal
ax0.plot(time_data, noisy_p, label="noisy signal")

# plot processed signal
ax0.plot(time_data, denoised_signal, label="denoised signal")

# plot saccade begin and end
# TODO

plt.plot()
plt.legend()
plt.show()
