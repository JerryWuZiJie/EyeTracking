import scipy.io
import matplotlib.pyplot as plt

import process_data


PATIENT_NUM = 1     # patient number, between 1 and 21
CONTROL = True      # True: control group, False: patient group


assert 1 <= PATIENT_NUM <= 21
patient_num = PATIENT_NUM - 1

# read in data
data = scipy.io.loadmat("data/antisaccadeContPatient.mat")

# patient data
sacc_patient = data['Dcell']
# target data
sacc_target = data['Tcell']

if CONTROL:
    # control group
    sacc_patient_group = sacc_patient[0, 0]
    sacc_target_group = sacc_target[0, 0]
else:
    # patient group
    sacc_patient_group = sacc_patient[0, 1]
    sacc_target_group = sacc_target[0, 1]

# actual data
sacc_patient_data = sacc_patient_group[patient_num, 0]
sacc_target_data = sacc_target_group[patient_num, 0]

### patient data ###
# horizontal eye movement
horizontal_data = sacc_patient_data[:, 0]
# vertical eye movement, don't care
_ = sacc_patient_data[:, 1]
# time data
time_data = sacc_patient_data[:, 2]

### target data ###
# horizontal eye movement
horizontal_target = sacc_target_data[:, 0]
# vertical eye movement, don't care
_ = sacc_target_data[:, 1]

Fs = 500  # TODO: need to calculate sampling frequency from data


denoised_signal, detection_array, total_sacs = process_data.process_data(
    Fs, horizontal_data)

vertical_line = process_data.sacc_start_end(detection_array)

### create figure ###
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.set_ylabel("Position (deg)")
ax0.set_xlabel("Time (s)")
ax0.set_title('CGTV: %d saccades detected' % total_sacs)

# plot detection
ax0.plot(time_data, detection_array*10, label="detection")

# plot detection start and end
for x in time_data[vertical_line == 1]:
    ax0.axvline(x=x, color='gray', alpha=0.5)
for x in time_data[vertical_line == -1]:
    ax0.axvline(x=x, color='gray', alpha=0.5, ls='--')

# plot original signal
ax0.plot(time_data, horizontal_data, label="noisy signal")

# plot processed signal
ax0.plot(time_data, denoised_signal, label="denoised signal")

# plot target
ax0.plot(time_data, horizontal_target, label="target")

plt.plot()
plt.legend()
plt.show()
