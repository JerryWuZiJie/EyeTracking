from __future__ import division
import numpy as np
from scipy import signal
from scipy import sparse
from scipy.sparse import linalg as slin
import os
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image
from saccade_model import saccade_model
import algorithms


NIT_MAX = 20                        # max iteration for denoising algorithm
IDLING = True
Fs = 500.0                          # sampling rate
FONT1 = ('times', 18)               # text font
FONT2 = ('times', 15, 'italic')     # parameters font
V_UPPER = 800                       # upper bound of velocity graph
V_LOWER = -50                       # lower bound of velocity graph
ALPHA_MAX = 15
BETA_MAX = 20
ALPHA_COEFF = (0.016 * Fs) if (Fs <= 500) else (0.0032*Fs + 6.4)
BETA_COEFF = (0.008*Fs) if (Fs <= 500) else (0.0016*Fs + 3.2)


def restart_program():
    """
    Restart the program.

    Will not able to restart if directory path contains space
    """
    python = sys.executable
    os.execl(python, python, * sys.argv)


def diff(x):
    """
    central differenece filter
    Calculate derivative.
    """
    h = np.array([0.5, 0, -0.5])
    y = signal.convolve(x, Fs*h, mode='same')
    y[0] = 0
    y[-1] = 0
    return y


def rmse(x, y):
    """
    Calculate root-mean-square-error.
    """
    z = np.sqrt(np.mean((x-y)**2))
    return z


def make_saccade(event):
    """
    Generate multiple saccades.
    """
    sacc_eta = eta.get()
    sacc_c = c.get()
    sacc_amp = amplitude.get()
    sacc_sigma = sigma.get()
    sacc_n = int(n.get())
    global t, s, y, w, x, prev_alpha, prev_beta, sacc_dur

    # generate clean saccade(s)
    waveform, velocity, peak_velocity = saccade_model(
        T, sacc_eta, sacc_c, sacc_amp)
    s = waveform
    if sacc_n > 1:
        for i in range(1, sacc_n):
            # start second saccades from negative position then flip it to concat
            waveform, velocity, peak_velocity = saccade_model(
                T, sacc_eta, sacc_c, sacc_amp, s0=(-1)**i*s[-1])
            s = np.concatenate((s, (-1)**i*waveform))

    N = len(s)
    t = np.arange(N)/Fs
    # update noise
    if len(w) != N:
        w = np.random.randn(N)
    # add noise to clean data
    y = s + w * sacc_sigma
    # update denoised data x
    x = y
    # calculate velocity (derivative)
    sd1 = diff(s)
    yd1 = diff(y)

    # update plots
    line1_n.set_data(t, y)
    p1.set_xlim((0, (N-1)/Fs))
    line2_n.set_data(t, yd1)
    p2.set_xlim((0, (N-1)/Fs))
    if sacc_n > 1:
        p2.set_ylim((-V_UPPER, V_UPPER))
    else:
        p2.set_ylim((V_LOWER, V_UPPER))
    line1_c.set_data(t, s)  # TODO: how to determine if drawing this?
    line2_c.set_data(t, sd1)
    fig.canvas.draw()

    # denoising
    sacc_dur = np.sum(abs(sd1) > 30)/Fs/sacc_n
    alpha = ALPHA_COEFF*sacc_sigma
    beta = BETA_COEFF*np.sqrt(sacc_amp)*np.exp(5*sacc_dur)*sacc_sigma
    lam1.set(alpha)
    lam2.set(beta)
    denoise(3)

    # clear prev_alpha to run more iteration later when idle
    prev_alpha = 0
    prev_beta = 0


def denoise_cb(event):
    # callback function for alpha and beta scale
    denoise(3)


def denoise(Nit):
    """
    Run CGTV to denoise the data.
    """
    alpha = lam1.get()
    beta = lam2.get()
    global x

    # denosing
    x = algorithms.cgtv(y, alpha, beta, Nit, x)
    
    xd1 = diff(x)
    line1_d.set_data(t, x)
    line2_d.set_data(t, xd1)
    err = rmse(x, s)
    p1.set_title('Simulated Eye Movement Data (RMSE = %.4f)' % err)
    fig.canvas.draw()


def denoiseDefault():
    """
    Use default parameters for CGTV to denoise the data.
    """
    sacc_amp = amplitude.get()
    sacc_sigma = sigma.get()
    alpha = ALPHA_COEFF*sacc_sigma
    beta = BETA_COEFF*sacc_sigma*np.sqrt(sacc_amp)*np.exp(5*sacc_dur)
    lam1.set(alpha)
    lam2.set(beta)
    denoise(NIT_MAX)


def new_noise():
    """
    Generate new noise realization.
    """
    global w, y, prev_alpha, prev_beta
    sacc_sigma = sigma.get()
    w = np.random.randn(len(s))
    y = s + w * sacc_sigma

    yd1 = diff(y)
    line1_n.set_data(t, y)
    line2_n.set_data(t, yd1)

    denoise(3)

    # clear prev_alpha to run more iteration later when idle
    prev_alpha = 0
    prev_beta = 0

    fig.canvas.draw()


def show_raw():
    """
    Show noise-free data.
    """
    if raw.get() == 1:
        line1_c.set_alpha(1)
        line2_c.set_alpha(1)
    else:
        line1_c.set_alpha(0)
        line2_c.set_alpha(0)
    fig.canvas.draw()


def switchStatus0(event):
    """
    Turn off background computing.
    """
    global IDLING
    IDLING = True


def switchStatus1(event):
    """
    Turn on background computing.
    """
    global IDLING
    IDLING = False


def denoise_conti():
    """
    Continuously checking if denoise parameter change. If changed, run denoise
    algorithm
    """
    root.after(1000, denoise_conti)
    if IDLING:
        return

    alpha = lam1.get()
    beta = lam2.get()
    global prev_alpha, prev_beta
    if alpha != prev_alpha or beta != prev_beta:
        prev_alpha = alpha
        prev_beta = beta
        denoise(NIT_MAX)


EPS = 1E-10                # smoothed penalty function
def psi(x): return np.sqrt(x**2 + EPS)


root = tk.Tk()
root.title('Saccade Denoising Demo')

# Define variables
eta = tk.DoubleVar(value=600.0)         # saccade parameter, eta
c = tk.DoubleVar(value=6.0)             # saccade parameter, c
amplitude = tk.DoubleVar(value=20.0)    # saccade amplitude
sigma = tk.DoubleVar(value=0.1)         # noise parameter
raw = tk.IntVar()                       # display raw data or not
lam1 = tk.DoubleVar(value=0)            # denoising parameter 1
lam2 = tk.DoubleVar(value=0)            # denoising parameter 2
n = tk.StringVar(value='1')             # number of saccades

### Drop down menu ###
myMenu = tk.Menu(root)
root.config(menu=myMenu)
subMenu_1 = tk.Menu(myMenu)
myMenu.add_cascade(label='Menu', menu=subMenu_1)
subMenu_1.add_command(label='Restart', command=restart_program)
subMenu_1.add_separator()
subMenu_1.add_command(label='Close', command=root.quit)

### Frames ###
rightFrame = tk.Frame(root)
rightFrame.pack(side='right', expand=0)
topFrame = tk.Frame(root)
topFrame.pack(side='top', fill='x', expand=0)

### Title ###
title = tk.Label(topFrame, text='Saccade Denoising Demo',
                 font=('times', 24, 'bold'))
title.pack(side='top')

### Left frame: plots ###
T = np.arange(-0.15, 0.15+1.0/Fs, 1.0/Fs)
waveform, velocity, peak_velocity = saccade_model(T, 600, 6, 10)
w = np.array([0])  # white noise

# Position plot
fig = matplotlib.figure.Figure()
p1 = fig.add_subplot(211)
line1_n, = p1.plot(T, waveform, color='k', linewidth=1.8, label='Noisy')
line1_d, = p1.plot(T, waveform, color='r', linewidth=1.5, label='Denoised')
line1_c, = p1.plot(T, waveform, color='b', alpha=0,
                   linewidth=1.5)  # clean data
p1.set_xlim((-0.15, 0.15))
p1.set_ylim((-1, 30))  # TODO: max_amp
p1.set_ylabel('Position (deg)')
p1.set_title('Simulated Eye Movement Data')
p1.legend(loc='upper left')

# Velocity plot
p2 = fig.add_subplot(212)
line2_n, = p2.plot(T, velocity, color='k', linewidth=1.8, label='Noisy')
line2_d, = p2.plot(T, waveform, color='r', linewidth=1.5, label='Denoised')
line2_c, = p2.plot(T, velocity, color='b', alpha=0, linewidth=1.5)
p2.set_xlim((-0.15, 0.15))
p2.set_ylim((V_LOWER, V_UPPER))
p2.set_xlabel('Time (s)')
p2.set_ylabel('Velocity (deg/s)')
p2.legend(loc='upper left')

# canvas for matplotlib plotting
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

### Right frame: parameters ###
# Saccade parameters
label_sp = tk.Label(rightFrame, text='Saccade Parameters:', font=FONT1)
label_sp.grid(row=0, column=1, columnspan=4, sticky='w')
# eta
paramLabel_eta = tk.Label(rightFrame, text=u'\u03b7', font=FONT2)
paramLabel_eta.grid(row=1, column=0, sticky='e')
paramScale_eta = tk.Scale(rightFrame, orient='horizontal', length=200,
                          variable=eta, from_=200, to=800, command=make_saccade)
paramScale_eta.grid(row=1, column=1, columnspan=3)
# c
paramLabel_c = tk.Label(rightFrame, text='c', font=FONT2)
paramLabel_c.grid(row=2, column=0, sticky='e')
paramScale_c = tk.Scale(rightFrame, orient='horizontal', length=200,
                        variable=c, from_=2, to=12, resolution=0.2,
                        command=make_saccade)
paramScale_c.grid(row=2, column=1, columnspan=3)
# A
paramLabel_A = tk.Label(rightFrame, text='Amplitude', font=FONT2)
paramLabel_A.grid(row=3, column=0, sticky='e')
paramScale_A = tk.Scale(rightFrame, orient='horizontal', length=200,
                        variable=amplitude, from_=1, to=30, resolution=0.5,
                        command=make_saccade)
paramScale_A.grid(row=3, column=1, columnspan=3)
# number of saccades
paramLabel_n = tk.Label(rightFrame, text='# Saccades', font=FONT2)
paramLabel_n.grid(row=4, column=0, sticky='e')
paramDropMenu_n = tk.OptionMenu(
    rightFrame, n, '1', '2', '3', '4', command=make_saccade)
paramDropMenu_n.config(width=15, bg='gray')
paramDropMenu_n.grid(row=4, column=1, columnspan=3)
# Noise parameter
label_np = tk.Label(rightFrame, text='Noise Parameter:', font=FONT1)
label_np.grid(row=5, column=1, columnspan=4, sticky='w')
paramLabel_sigma = tk.Label(rightFrame, text=u'\u03c3', font=FONT2)
paramLabel_sigma.grid(row=6, column=0, sticky='e')
paramScale_sigma = tk.Scale(rightFrame, orient='horizontal', length=200,
                            variable=sigma, from_=0, to=1, resolution=0.02,
                            command=make_saccade)  # TODO: change to new_noise?
paramScale_sigma.grid(row=6, column=1, columnspan=3)
paramScale_sigma.set(0.5)
button_noise = tk.Button(rightFrame, text='Update noise',
                         width=20, height=2, command=new_noise)
button_noise.grid(row=7, column=1, columnspan=3)

# Show noise-free data
checkbox_clean_data = tk.Checkbutton(
    rightFrame, text='Show clean data', variable=raw, command=show_raw)
checkbox_clean_data.grid(row=8, column=1, columnspan=3)

# Denoising Parameters
label_dp = tk.Label(rightFrame, text='Denoising Parameters:', font=FONT1)
label_dp.grid(row=9, column=1, columnspan=4, sticky='w')
# cost function
label_cost = tk.Label(rightFrame)
label_cost.grid(row=10, column=0, columnspan=2)
formula = matplotlib.figure.Figure(figsize=(3, 0.5))
ax_f = formula.add_subplot(111)
ax_f.set_axis_off()
ax_f.text(
    -0.1, 0.3, "$x=\\arg\ \min_x \{0.5 \Vert y-x \Vert _2^2 + \
    \\alpha \Vert D_1 x \Vert_1 + \\beta\Vert D_3 x\Vert_1\}$", fontsize=9)
canvas.draw()
canvas_f = FigureCanvasTkAgg(formula, master=label_cost)
canvas_f.get_tk_widget().pack(side="left", fill="x", expand=True)
# alpha
paramLabel_alpha = tk.Label(rightFrame, text=u'\u03b1', font=FONT2)
paramLabel_alpha.grid(row=11, column=0, sticky='e')
paramScale_alpha = tk.Scale(rightFrame, orient='horizontal', command=denoise_cb,
                            length=200, variable=lam1, from_=0, to=ALPHA_MAX,
                            resolution=0.05)
paramScale_alpha.grid(row=11, column=1, columnspan=3)
# beta
paramLabel_beta = tk.Label(rightFrame, text=u'\u03b2', font=FONT2)
paramLabel_beta.grid(row=12, column=0, sticky='e')
paramScale_beta = tk.Scale(rightFrame, orient='horizontal', command=denoise_cb,
                           length=200, variable=lam2, from_=0, to=BETA_MAX,
                           resolution=0.05)
paramScale_beta.grid(row=12, column=1, columnspan=3)
# Default parameters button
button_params = tk.Button(rightFrame, text='Default Parameters',
                          width=20, height=2, command=denoiseDefault)
button_params.grid(row=13, column=1, columnspan=3)

# whitespace between buttons
whitespace = tk.Label(rightFrame, text=' ')
whitespace.grid(row=19, column=0)
# exit button
button_exit = tk.Button(rightFrame, text='Exit',
                        width=20, height=2, command=root.quit)
button_exit.grid(row=20, column=1, columnspan=3)

label_reference = tk.Label(
    root, text='"Detection of normal and slow saccades using implicit piecewise polynomial approximation" \nby W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker, and T. Hudson', justify=tk.LEFT)
label_reference.pack(side=tk.LEFT)

root.bind("<ButtonPress>", switchStatus0)
root.bind("<ButtonRelease>", switchStatus1)

# program start here
prev_alpha = 0
prev_beta = 0
denoise_conti()

root.mainloop()
