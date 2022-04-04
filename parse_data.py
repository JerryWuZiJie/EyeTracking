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
import scipy.signal
import scipy.optimize
import scipy.stats

import denoise_algo

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
DUR_TH_N = int(DUR_TH * Fs)     # duration threshold (samples)
FIX_TH = 0.04                   # fixation threshold (s)
FIX_TH_N = int(FIX_TH * Fs)     # fixation threshold (samples)

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

### calculate white noise (sigma) ###
# get velocity by difference filter
est_v = diff(noisy_p)
# use moving average to smooth the data (could also use low pass filter)
est_v_smooth = np.convolve(est_v, np.ones(MOVAVG_FS), 'same') / MOVAVG_FS
# use velocity and duration threshold to get array of 1 and 0's indicating saccades
detection_array = np.zeros_like(est_v_smooth)  # 1 for saccade, 0 for fixation
dur_counter = 0
est_total_sac = 0
for i in range(len(est_v_smooth)):
    if np.abs(est_v_smooth[i]) >= V_TH:
        # if greater than threshold, it's a saccade
        detection_array[i] = 1
        if dur_counter == 0:
            # add one more saccade when first vt is exceeded
            est_total_sac += 1
        dur_counter += 1
    else:
        detection_array[i] = 0
        if dur_counter > 0 and dur_counter < DUR_TH_N:
            # if duration is less than threshold, it's a fixation
            detection_array[i-dur_counter:i] = 0
            est_total_sac -= 1
        dur_counter = 0
# process the array to remove short fixation
fix_counter = 0
for i in range(len(detection_array)):
    if detection_array[i] == 0:
        fix_counter += 1
    else:
        if fix_counter > 0 and fix_counter <= FIX_TH_N:
            # fixation between two saccades is short, count it as saccades
            detection_array[i-fix_counter:i] = 1
            est_total_sac -= 1  # 2 merge to 1
        fix_counter = 0
# get standard deviation for all fixations, this will be the signma
# TODO subtract average for each of saccades
est_sigma = scipy.stats.tstd(noisy_p[detection_array == 0])
# get average duration
est_total_dur = (detection_array == 1).sum()
est_dur = est_total_dur / est_total_sac / Fs
# get average amplitude TODO: seems wrong, check saccade_cgtv 83 amp_avg?
est_total_amp = np.abs(est_v_smooth[detection_array == 1]).sum()
est_amp = est_total_amp / est_total_sac / Fs
# calculate sigma using array based on Fs, avg_amp, avg_dur, and sigma
if Fs <= 500:
    alpha = 0.016 * Fs * est_sigma
    beta = 0.008 * Fs * np.sqrt(est_amp) * np.exp(5 * est_dur)
else:
    alpha = (0.0032*Fs + 6.4) * est_sigma
    beta = (0.0016*Fs + 3.2) * np.sqrt(est_amp) * np.exp(5 * est_dur)

# denoise signal
denoised_signal = denoise_algo.cgtv(noisy_p, alpha, beta, ITER_N)


# create figure
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.set_ylabel("Position (deg)")
ax0.set_xlabel("Time (s)")
# TODO below: change 10 to number of saccades
ax0.set_title('CGTV: %d saccades detected' % 10)

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
