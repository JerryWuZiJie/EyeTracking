import scipy.io
import matplotlib.pyplot as plt

import process_data

data = scipy.io.loadmat("antisaccadeContPatient.mat")

# first time series of the control group
first_of_control = data['Dcell'][0, 0][0, 0]

x, y, time_data = first_of_control[:,
                                   0], first_of_control[:, 1], first_of_control[:, 2]
Fs = 500  # TODO: need to calculate sampling frequency from data


denoised_signal, detection_array, total_sacs = process_data.process_data(
    Fs, x)

# create figure
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.set_ylabel("Position (deg)")
ax0.set_xlabel("Time (s)")
ax0.set_title('CGTV: %d saccades detected' % total_sacs)

# plot detection
ax0.plot(time_data, detection_array*10, label="detection")

# plot original signal
ax0.plot(time_data, x, label="noisy signal")

# plot processed signal
ax0.plot(time_data, denoised_signal, label="denoised signal")

plt.plot()
plt.legend()
plt.show()
