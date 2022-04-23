import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import process_data

# parameter used for simulation data
SIGMA = 0.1     # noise standard deviation
Fs = 500        # sampling rate

# set random seed
np.random.seed(0)

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


w = np.random.randn(*signal_data.shape)
noisy_p = signal_data + w * SIGMA

denoised_signal, detection_array, total_sacs = process_data.process_data(
    Fs, noisy_p)


# create figure
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.set_ylabel("Position (deg)")
ax0.set_xlabel("Time (s)")
ax0.set_title('CGTV: %d saccades detected' % total_sacs)

# plot detection
ax0.plot(time_data, detection_array*10, label="detection")

# plot original signal
ax0.plot(time_data, noisy_p, label="original signal")

# plot processed signal
ax0.plot(time_data, denoised_signal, label="denoised signal")

plt.plot()
plt.legend()
plt.show()
