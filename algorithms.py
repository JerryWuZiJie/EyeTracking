from scipy import sparse
from scipy.sparse import linalg as slin
import scipy.signal
import numpy as np


def diff(x, Fs):
    """
    central differenece filter
    Calculate derivative.
    
    @param x: input signal
    @param Fs: sampling frequency
    
    @ret y: derivative signal
    """
    h = np.array([0.5, 0, -0.5])
    y = scipy.signal.convolve(x, Fs*h, mode='same')
    y[0] = 0
    y[-1] = 0
    return y


def v_denoise(position, Fs, movavg=0.02):
    """
    turn position into velocity and smooth out
    maybe use LPF in the future
    
    @param position: position signal
    @param Fs: sampling frequency
    @param movavg: moving average in units of seconds

    @ret vel_smooth: smoothed velocity signal
    """
    
    # get velocity by difference filter
    vel = diff(position, Fs)
    # use moving average to smooth the data (could also use low pass filter)
    movavg = int(movavg * Fs)
    vel_smooth = np.convolve(vel, np.ones(movavg), 'same') / movavg
    
    return vel_smooth


def VT(position, Fs, v_th, dur_th, fix_th, movavg=0.02):
    """
    Velocity Threshold algorithm

    @param position: position signal
    @param Fs: sampling frequency
    @param v_th: velocity threshold in unit of samples
    @param dur_th: duration threshold in unit of seconds
    @param fix_th: fixation threshold in unit of seconds
    @param movavg_n: moving average in units of seconds
    
    @ret detection_array: array of 1s and 0s, 1 for saccades and 0 for fixations
    @ret total_sacs: total saccades detected
    """

    # get smooth velocity signal
    vel_smooth = v_denoise(position, Fs, movavg)
    # use velocity and duration threshold to get array of 1s and 0s
    # 1 for saccade, 0 for fixation
    detection_array = np.zeros_like(vel_smooth)
    dur_counter = 0
    total_sacs = 0
    dur_th_n = int(dur_th * Fs)  # in unit of samples
    for i in range(len(vel_smooth)):
        if np.abs(vel_smooth[i]) >= v_th:
            # if greater than threshold, it's a saccade
            detection_array[i] = 1
            if dur_counter == 0:
                # add one more saccade when first vt is exceeded
                total_sacs += 1
            dur_counter += 1
        else:
            detection_array[i] = 0
            if dur_counter > 0 and dur_counter < dur_th_n:
                # if duration is less than threshold, it's a fixation
                detection_array[i-dur_counter:i] = 0
                total_sacs -= 1
            dur_counter = 0
    # remove short fixation if it's less than fixation threshold
    fix_counter = 0
    fix_th_n = int(fix_th * Fs)  # in unit of samples
    for i in range(len(detection_array)):
        if detection_array[i] == 0:
            fix_counter += 1
        else:
            if fix_counter > 0 and fix_counter <= fix_th_n:
                # fixation between two saccades is short, count it as saccades
                detection_array[i-fix_counter:i] = 1
                total_sacs -= 1  # 2 saccades merge to 1
            fix_counter = 0
            
    return detection_array, total_sacs


def main_squence(x_amp, eta, c):
    """
    formula for main sequence

    xdata: input
    V = eta*(1-e^(-A/c))
    """

    return eta * (1-np.exp(-x_amp/c))


EPS = 1E-10                # smoothed penalty function
def psi(x): return np.sqrt(x**2 + EPS)


def cgtv(noisy_signal, alpha, beta, Nit, denoised_signal=None):
    """
    run the algorithm

    noisy_signal: original position signal
    alpha: denoising parameter
    beta: denoising parameter
    Nit: number of iteration
    denoised_signal: a previously denoised signal
    """

    N = len(noisy_signal)
    e = np.ones(N)
    D1 = sparse.spdiags([-e, e], [0, 1], N-1, N)
    D3 = sparse.spdiags([-e, 3*e, -3*e, e], [0, 1, 2, 3], N-3, N)
    I = sparse.spdiags(e, 0, N, N)

    # if previously denoised signal exists, continue denosing on that signal
    if denoised_signal is None:
        x = noisy_signal
    else:
        x = denoised_signal

    # run algorithm (check paper for detail)
    for i in range(Nit):
        Lam1 = sparse.spdiags(alpha/psi(np.diff(x)), 0, N-1, N-1)
        Lam3 = sparse.spdiags(beta/psi(np.diff(x, 3)), 0, N-3, N-3)
        temp = I + ((D1.T).dot(Lam1)).dot(D1) + ((D3.T).dot(Lam3)).dot(D3)
        x = slin.spsolve(temp, noisy_signal)
    return x
