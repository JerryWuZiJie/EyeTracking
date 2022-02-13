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


def restart_program():
    """
    Restart the program.
    """
    python = sys.executable
    os.execl(python, python, * sys.argv)
#    os.execl(python, python, * ['Saccade_ver2.py'])
  
def make_saccade(event):
    """
    Generate a saccade.
    """      
    sacc_eta = eta.get()
    sacc_c = c.get()
    sacc_amp = amplitude.get()

    waveform, velocity, peak_velocity = saccade_model(t, sacc_eta, sacc_c, sacc_amp)

    line1.set_data(t, waveform)
    # p1.set_ylim((-2,np.ceil(sacc_amp/10)*10+2))
    p1.set_ylim((-2, 32))
    
    line_ms.set_ydata(sacc_eta*(1-np.exp(-a/sacc_c)))
    dot_ms.set_data(sacc_amp, peak_velocity)

    fig.canvas.draw()    
 
root = tk.Tk()
root.title('A Parametric Saccade Model')
# root.geometry('1420x580')
#root.resizable(width=False, height=False)

# Define variables
eta = tk.DoubleVar(value=600.0)         # saccade parameter, eta
c = tk.DoubleVar(value=6.0)             # saccade parameter, c
amplitude = tk.DoubleVar()    # saccade amplitude

### Drop down menu ###
myMenu = tk.Menu(root)
root.config(menu=myMenu)
subMenu_1 = tk.Menu(myMenu)
myMenu.add_cascade(label='Menu', menu=subMenu_1)
subMenu_1.add_command(label='Restart', command=restart_program)
subMenu_1.add_separator()
subMenu_1.add_command(label='Close', command=root.quit)

### Frames ###
#status = tk.Label(root, text='Change parameters to check the corresponding results', bd=1, relief='sunken', anchor='w') 
# status = tk.Label(root, text='"A parametric model for saccadic eye movement". IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016. DOI: 10.1109/SPMB.2016.7846860', bd=1, relief='sunken', anchor='w', wraplength = 600, justify = tk.LEFT) 
status = tk.Label(root, text='"A parametric model for saccadic eye movement". IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016. DOI: 10.1109/SPMB.2016.7846860', bd=1, relief='sunken', anchor='w', wraplength = 600, justify = tk.LEFT) 
status.pack(side='bottom', fill='x')
topFrame = tk.Frame(root, height=200)
topFrame.pack(side='top', fill='x')
# leftFrame = tk.Frame(root)
# leftFrame.pack(side='left')
rightFrame = tk.Frame(root)
rightFrame.pack(side='right')
# rightFrame.pack(side='left')

### Top frame ###
title = tk.Label(topFrame, text='A Parametric Model for Saccadic Waveforms', font=('times', 30, 'bold'))
title.pack(side='top')

### Left frame ###
Fs = 500.0
t = np.arange(-0.15, 0.15+1.0/Fs, 1.0/Fs)
waveform, velocity, peak_velocity = saccade_model(t, 600, 6, 10)
LW = 2
MS = 16
# Position plot
# fig = matplotlib.figure.Figure(figsize=(11,4))
fig = matplotlib.figure.Figure(figsize=(9,4))
# fig = matplotlib.figure.Figure()
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(left=0.07, right = 0.95, bottom = 0.15)

# fig.subplots_adjust(hspace=0.5)   # height between subplots. not needed here.
p1 = fig.add_subplot(1, 2, 1)
line1, = p1.plot(t, waveform, color='k', linewidth=LW)
p1.set_xlim((-0.15,0.15))
p1.set_ylim((-2,12))
p1.set_xlabel('Time (sec)', fontsize=12)
p1.set_ylabel('Angle (deg)', fontsize=12)
p1.set_title('Saccade waveform', fontsize=14)
# Main sequence plot
p2 = fig.add_subplot(1, 2, 2)
a = np.arange(0, 35, 0.01)
line_ms, = p2.plot(a, 600*(1-np.exp(-a/6)), color='b', linewidth=LW)
dot_ms, = p2.plot(10, peak_velocity, 'k.', markersize=MS)
p2.set_xticks([0,10,20,30])
p2.set_xticklabels([0,10,20,30])
p2.set_xlim((0,32))
p2.set_yticks([0,200,400,600,800])
p2.set_yticklabels([0,200,400,600,800])
p2.set_ylim((0,820))
p2.set_xlabel('Amplitude (deg)', fontsize=12)
p2.set_ylabel('Peak velocity (deg/s)', fontsize=12)
p2.set_title('Saccade main sequence', fontsize=14)
p2.legend((line_ms, ),('$\eta(1-e^{-A/c})$', ), loc='lower right')

# canvas = FigureCanvasTkAgg(fig, master=leftFrame)
canvas = FigureCanvasTkAgg(fig, master = root)
W1 = canvas.get_tk_widget()
# W1.pack()
W1.pack(fill = tk.BOTH, expand = 1)


### Right frame ###
font1 = ('times',20)
font2 = ('times',15,'italic')
# formulae
label_s = tk.Label(rightFrame, text='Saccade model formula:', font=font1)
label_s.grid(row=0, column=0, columnspan=5)
photo_s = Image.open('fun_s.png')
fun_s = ImageTk.PhotoImage(photo_s)
label_fun_s = tk.Label(rightFrame, image=fun_s)
label_fun_s.grid(row=1, columnspan=5)
photo_f = Image.open('fun_f.png')
fun_f = ImageTk.PhotoImage(photo_f)
label_fun_f = tk.Label(rightFrame, image=fun_f)
label_fun_f.grid(row=2, columnspan=5)

label_7 = tk.Label(rightFrame, text=' ')
label_7.grid(row=3, column=0)

# Saccade parameters
label_1 = tk.Label(rightFrame, text='Saccade Parameters:', font=font1)
label_1.grid(row=5, column=0, columnspan=5)
paramLabel_eta = tk.Label(rightFrame, text=u'\u03b7', font=font2)
paramLabel_eta.grid(row=6, column=0, sticky='e')
paramScale_eta = tk.Scale(rightFrame, orient='horizontal', length = 200, variable=eta, from_=200, to=800, resolution=0.5, command=make_saccade)
paramScale_eta.grid(row=6, column=1, columnspan=3)
paramLabel_c = tk.Label(rightFrame, text='c', font=font2)
paramLabel_c.grid(row=7, column=0, sticky='e')
paramScale_c = tk.Scale(rightFrame, orient='horizontal', length = 200, variable=c, from_=2, to=12, resolution=0.1, command=make_saccade)
paramScale_c.grid(row=7, column=1, columnspan=3)
paramLabel_A = tk.Label(rightFrame, text='A', font=font2)
paramLabel_A.grid(row=8, column=0, sticky='e')
paramScale_A = tk.Scale(rightFrame, orient='horizontal', length = 200, variable=amplitude, from_=1, to=30, resolution=0.2, command=make_saccade)
paramScale_A.grid(row=8, column=1, columnspan=3)
paramScale_A.set(10.0)

label_8 = tk.Label(rightFrame, text=' ')
label_8.grid(row=19, column=0)

button_9 = tk.Button(rightFrame, text='Exit', width=20, height=2, command=root.quit)
button_9.grid(row=20, column=1, columnspan=3)
  
root.mainloop()
