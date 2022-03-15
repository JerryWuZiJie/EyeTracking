from __future__ import division
import numpy as np
import os
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from saccade_model import saccade_model

LW = 2  # line width of the plot
MS = 16  # marker size of the plot
FONT1 = ('times', 20)  # text font
FONT2 = ('times', 15, 'italic')  # parameters font
MAX_AMP = 35  # maximum amplitude of saccade in degrees


def restart_program():
    """
    Restart the program.

    Will not able to restart if directory path contains space
    """
    python = sys.executable
    os.execl(python, python, * sys.argv)


def make_saccade(event):
    """
    Generate a saccade.
    """
    sacc_eta = eta.get()
    sacc_c = c.get()
    sacc_amp = amplitude.get()

    waveform, velocity, peak_velocity = saccade_model(
        t_xaxis, sacc_eta, sacc_c, sacc_amp)

    line1.set_data(t_xaxis, waveform)
    p1.set_ylim((-2, MAX_AMP))

    line_ms.set_ydata(sacc_eta*(1-np.exp(-amplitude_xaxis/sacc_c)))
    dot_ms.set_data(sacc_amp, peak_velocity)

    fig.canvas.draw()


root = tk.Tk()
root.title('A Parametric Saccade Model')

### Drop down menu ###
myMenu = tk.Menu(root)
root.config(menu=myMenu)
subMenu_1 = tk.Menu(myMenu)
myMenu.add_cascade(label='Menu', menu=subMenu_1)
subMenu_1.add_command(label='Restart', command=restart_program)
subMenu_1.add_separator()
subMenu_1.add_command(label='Close', command=root.quit)

### Frames ###
titleFrame = tk.Frame(root, height=200)
titleFrame.pack(side='top', fill='x')
rightFrame = tk.Frame(root)
rightFrame.pack(side='right')

### Title frame ###
# title
title = tk.Label(titleFrame, text='A Parametric Model for Saccadic Waveforms',
                 font=('times', 30, 'bold'))
title.pack(side='top')

### Left frame: plots ###
# Position plot
fig = matplotlib.figure.Figure(figsize=(9, 4))
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15)

# Saccade waveform plot
p1 = fig.add_subplot(121)
Fs = 500.0  # sample rate
t_xaxis = np.arange(-0.15, 0.15+1.0/Fs, 1.0/Fs)  # x axis: time
waveform, velocity, peak_velocity = saccade_model(t_xaxis, 600, 6, 10)
line1, = p1.plot(t_xaxis, waveform, color='k', linewidth=LW)
p1.set_xlim((-0.15, 0.15))
p1.set_ylim((-2, 12))
p1.set_xlabel('Time (sec)', fontsize=12)
p1.set_ylabel('Angle (deg)', fontsize=12)
p1.set_title('Saccade waveform', fontsize=14)

# Saccade main sequence plot
p2 = fig.add_subplot(122)
amplitude_xaxis = np.arange(0, MAX_AMP, 0.01)
line_ms, = p2.plot(amplitude_xaxis, 600 *
                   (1-np.exp(-amplitude_xaxis/6)), color='b', linewidth=LW)
dot_ms, = p2.plot(10, peak_velocity, 'k.', markersize=MS)
p2.set_xticks([0, 10, 20, 30])
p2.set_xticklabels([0, 10, 20, 30])
p2.set_xlim((0, MAX_AMP))
p2.set_yticks([0, 200, 400, 600, 800])
p2.set_yticklabels([0, 200, 400, 600, 800])
p2.set_ylim((0, 820))
p2.set_xlabel('Amplitude (deg)', fontsize=12)
p2.set_ylabel('Peak velocity (deg/s)', fontsize=12)
p2.set_title('Saccade main sequence', fontsize=14)
p2.legend((line_ms, ), ('$\eta(1-e^{-A/c})$', ), loc='lower right')

# canvas for matplotlib plotting
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=1)

### Right frame ###
# formulae
label_smf = tk.Label(rightFrame, text='Saccade model formula:', font=FONT1)
label_smf.grid(row=0, column=0, columnspan=5)

# s(t) formula
photo_s = Image.open('fun_s.png')
fun_s = ImageTk.PhotoImage(photo_s)
label_fun_s = tk.Label(rightFrame, image=fun_s)
label_fun_s.grid(row=1, columnspan=5)

# f(t) formula
photo_f = Image.open('fun_f.png')
fun_f = ImageTk.PhotoImage(photo_f)
label_fun_f = tk.Label(rightFrame, image=fun_f)
label_fun_f.grid(row=2, columnspan=5)

# whitespace between formulae and parameters
white_space = tk.Label(rightFrame, text=' ')
white_space.grid(row=3, column=0)

# Saccade parameters
# Define variables
eta = tk.DoubleVar(value=600.0)         # saccade parameter, eta
c = tk.DoubleVar(value=6.0)             # saccade parameter, c
amplitude = tk.DoubleVar()    # saccade amplitude
# text
label_sp = tk.Label(rightFrame, text='Saccade Parameters:', font=FONT1)
label_sp.grid(row=5, column=0, columnspan=5)
# eta
paramLabel_eta = tk.Label(rightFrame, text=u'\u03b7', font=FONT2)
paramLabel_eta.grid(row=6, column=0, sticky='e')
paramScale_eta = tk.Scale(rightFrame, orient='horizontal', length=200,
                          variable=eta, from_=200, to=800, resolution=0.5,
                          command=make_saccade)
paramScale_eta.grid(row=6, column=1, columnspan=3)
# c
paramLabel_c = tk.Label(rightFrame, text='c', font=FONT2)
paramLabel_c.grid(row=7, column=0, sticky='e')
paramScale_c = tk.Scale(rightFrame, orient='horizontal', length=200,
                        variable=c, from_=2, to=12, resolution=0.1,
                        command=make_saccade)
paramScale_c.grid(row=7, column=1, columnspan=3)
# A
paramLabel_A = tk.Label(rightFrame, text='A', font=FONT2)
paramLabel_A.grid(row=8, column=0, sticky='e')
paramScale_A = tk.Scale(rightFrame, orient='horizontal', length=200,
                        variable=amplitude, from_=1, to=30, resolution=0.2,
                        command=make_saccade)
paramScale_A.grid(row=8, column=1, columnspan=3)
paramScale_A.set(10.0)

# whitespace between parameters and exit button
whitespace2 = tk.Label(rightFrame, text=' ')
whitespace2.grid(row=19, column=0)

# exit button
button_exit = tk.Button(rightFrame, text='Exit', width=20,
                        height=2, command=root.quit)
button_exit.grid(row=20, column=1, columnspan=3)

# footnote
label_reference = tk.Label(root, text='"A parametric model for saccadic eye movement". '
                  'IEEE Signal Processing in Medicine and Biology Symposium '
                  '(SPMB), December 2016. DOI: 10.1109/SPMB.2016.7846860', bd=1,
                  relief='sunken', anchor='w', wraplength=600, justify=tk.LEFT)
label_reference.pack(side='bottom', fill='x')

root.mainloop()
