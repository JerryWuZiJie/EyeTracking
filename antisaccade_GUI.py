import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

import process_data


PATIENT_NUM_MIN = 1
PATIENT_NUM_MAX = 21


def change_data_disp():
    """
    change the display patient info
    """

    if patient_group:
        patient_label.configure(
            text="%s group: patient number: %d" % ('concussion', patient_num))
    else:
        patient_label.configure(
            text="%s group: patient number: %d" % ('control', patient_num))


def create_plot(fig):
    """
    plot the data to figure

    @param fig: matplotlib figure object to plot to
    """

    # Saccade waveform plot

    # setup plot
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.set_ylabel("Position (deg)")
    ax0.set_xlabel("Time (s)")
    ax0.set_title('CGTV: %d saccades detected' % total_sacs, fontsize=14)

    # plot detection
    for data_key in plot_data_dict:
        # if visible, plot it
        if plot_data_dict[data_key]['visible_var'].get():
            ax0.plot(*plot_data_dict[data_key]['params'], label=data_key)

    # This enables ploting vertical line, but doesn't fit with the work flow
    # # plot detection start and end
    # for x in time_data[vertical_line == 1]:
    #     ax0.axvline(x=x, color='gray', alpha=0.5)
    # for x in time_data[vertical_line == -1]:
    #     ax0.axvline(x=x, color='gray', alpha=0.5, ls='--')

    # show legend
    ax0.legend()


def update_plot():
    """
    update the plot in GUI
    """

    ax0, = fig.axes
    ax0.remove()
    create_plot(fig)
    fig.canvas.draw()


def open_plot(event=None):
    """
    open a separate plot, so can use matplotlib tools
    """

    fig = plt.figure()
    create_plot(fig)
    plt.show()


def update_data():
    """
    update the data to corresponding group and patient
    """

    global total_sacs, plot_data_dict, vertical_line
    # patient data
    sacc_patient = data['Dcell']
    # target data
    sacc_target = data['Tcell']

    if patient_group:
        # patient group
        sacc_patient_group = sacc_patient[0, 1]
        sacc_target_group = sacc_target[0, 1]
    else:
        # control group
        sacc_patient_group = sacc_patient[0, 0]
        sacc_target_group = sacc_target[0, 0]

    # actual data
    sacc_patient_data = sacc_patient_group[patient_num-1, 0]
    sacc_target_data = sacc_target_group[patient_num-1, 0]

    ### patient data ###
    # horizontal eye movement
    horizontal_data = sacc_patient_data[:, 0]
    # # vertical eye movement, don't care
    # _ = sacc_patient_data[:, 1]
    # time data
    time_data = sacc_patient_data[:, 2]

    ### target data ###
    # horizontal eye movement
    horizontal_target = sacc_target_data[:, 0]
    # # vertical eye movement, don't care
    # _ = sacc_target_data[:, 1]

    # process data using CGTV and VT algorithm
    denoised_signal, detection_array, total_sacs = process_data.process_data(
        time_data, horizontal_data)

    vertical_line = process_data.sacc_start_end(detection_array)

    # list of data to be plotted
    plot_data_dict = {}
    plot_data_dict['detection'] = {'visible_var': tk_line0,
                                   'params': [time_data, detection_array, 'b']}
    plot_data_dict['original signal'] = {'visible_var': tk_line1,
                                         'params': [time_data, horizontal_data, 'r']}
    plot_data_dict['denoised signal'] = {'visible_var': tk_line2,
                                         'params': [time_data, denoised_signal, 'g']}
    plot_data_dict['target'] = {'visible_var': tk_line3,
                                'params': [time_data, horizontal_target, 'orange']}


def update_patient():
    """
    update all
    """

    # update dropdown menu select num
    patient_option_var.set(str(patient_num))
    # update patient display info
    change_data_disp()
    # calculate new data
    update_data()
    # plot new data
    update_plot()


def left_key(event=None):
    """
    switch to next patient data when left key is pressed or left arrow button 
    is clicked

    @param event: event object from tkinter
    """

    global patient_num
    patient_num -= 1
    if patient_num < PATIENT_NUM_MIN:
        patient_num = PATIENT_NUM_MIN
    else:
        update_patient()


def right_key(event=None):
    """
    switch to next patient data when right key is pressed or right arrow button 
    is clicked

    @param event: event object from tkinter
    """

    global patient_num
    patient_num += 1
    if patient_num > PATIENT_NUM_MAX:
        patient_num = PATIENT_NUM_MAX
    else:
        update_patient()


def change_patient(patient_num_str):
    """
    change patient when dropdown menu is changed
    """

    global patient_num
    if patient_num != int(patient_num_str):
        patient_num = int(patient_num_str)
        update_patient()


def change_group(patient_group_str):
    """
    change the group when button pressed
    """

    global patient_group
    if patient_group_str == "Control Group" and patient_group == 1:
        patient_group = 0
        update_patient()
    elif patient_group_str == "Concussion Group" and patient_group == 0:
        patient_group = 1
        update_patient()


# first patient info
patient_group = 0               # 0 for control group, 1 concussion group
patient_num = 1                 # between 1 and 21

# read in data
data = scipy.io.loadmat("data/antisaccadeContPatient.mat")
Fs = 500  # TODO: need to calculate sampling frequency from data


### create tkinter GUI ###
root = tk.Tk()
root.title('A Parametric Saccade Model')

# bind left right keys to switch patient
root.bind("<Left>", left_key)
root.bind("<Right>", right_key)

# create frames
top_frame = tk.Frame(root)
bottom_frame = tk.Frame(root)
tools_frame = tk.Frame(bottom_frame)
open_plot_frame = tk.Frame(bottom_frame)


### top frame widgets ###
patient_label = tk.Label(top_frame, text="", font=('Arial', 20))
change_data_disp()  # change the display patient info for patient_label

patient_group_var = tk.StringVar(root, "Control Group")
patient_group_option = tk.OptionMenu(top_frame, patient_group_var,
                                  "Control Group", "Concussion Group",
                                  command=change_group)

patient_option_var = tk.StringVar(root, '1')
patient_option = tk.OptionMenu(top_frame, patient_option_var,
                               *[str(i+1) for i in range(PATIENT_NUM_MAX)],
                               command=change_patient)


### Middle frame: plots ###
# tk integer to control visibility of plot
tk_line0 = tk.IntVar(value=1)
tk_line1 = tk.IntVar(value=1)
tk_line2 = tk.IntVar(value=1)
tk_line3 = tk.IntVar(value=1)
# process data
update_data()

# Position plot
fig = matplotlib.figure.Figure(figsize=(9, 4))
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15)

# plot to figure
create_plot(fig)

# canvas widget for matplotlib and tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()


### Bottom frame widgets ###
# left right button to swtich patient
left_but = tk.Button(tools_frame, text='<-', command=left_key)
right_but = tk.Button(tools_frame, text='->', command=right_key)

# checkbuttons to show/hide certain data
display_but = []
for key in plot_data_dict:
    but = tk.Checkbutton(tools_frame, text=key,
                         variable=plot_data_dict[key]['visible_var'],
                         onvalue=1,
                         offvalue=0,
                         command=update_plot)
    display_but.append(but)

# button to open plot in separate matplotlib window
open_plot_but = tk.Button(open_plot_frame, text='open plot', command=open_plot)


### place frames ###
top_frame.pack()
canvas_widget.pack()
bottom_frame.pack(side='bottom')
tools_frame.pack()
open_plot_frame.pack()

### place other widgets ###
# top frame
patient_label.pack(side=tk.LEFT)
patient_group_option.pack(side=tk.LEFT)
patient_option.pack(side=tk.LEFT)
# bottom frame: tools
left_but.pack(side=tk.LEFT)
for but in display_but:
    but.pack(side=tk.LEFT)
right_but.pack(side=tk.LEFT)
# bottom frame: open plot
open_plot_but.pack()

root.mainloop()
