"""
process data and apply CGTV algorithms
"""

import numpy as np
import scipy.optimize
import scipy.stats

import algorithms


def process_data(time_data, original, V_TH=10, DUR_TH=0.024, FIX_TH=0.04, MOVAVG=0.02, ITER_N=20):
    """
    apply CGTV algorithms to original data

    FUTURE: plot main sequence
        np.nonzero https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
        scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        indexing: np.ix_(): https://stackoverflow.com/a/11393946/13720936
        INITIAL_FIT_VALS = [600, 8]     # intial guess value for curve_fit
        check saccade_cgtv.m for reference

    @param time_data: time data (s)
    @param original: original signal
    @param V_TH: velocity threshold (deg/s)
    @param DUR_TH: duration threshold (s)
    @param FIX_TH: fixation threshold (s)
    @param MOVAVG: move average filter (s)
    @param ITER_N: number of iteration times

    @return: denoised signal, detection array, total saccades
    """

    # calculate Fs from time data
    time_diff = np.diff(time_data)
    time_diff = sum(time_diff)/len(time_diff)
    Fs = round(1/time_diff)
    # run VT algorithm to get detection array, 1 for saccade and 0 for fixation
    est_detect_array, est_total_sac = algorithms.VT(
        original, Fs, V_TH, DUR_TH, FIX_TH, MOVAVG)
    # get standard deviation for all fixations, this will be the signma
    # subtract average for each fixation
    fix_len = 0  # len of each fixation
    fixations = np.copy(original)
    for i in range(len(est_detect_array)):
        if est_detect_array[i] == 0:
            fix_len += 1
        elif fix_len != 0:
            fixations[i-fix_len:i] -= np.mean(original[i-fix_len:i])
            fix_len = 0
    est_sigma = scipy.stats.tstd(fixations[est_detect_array == 0])
    # get average duration
    est_total_dur = (est_detect_array == 1).sum()
    est_dur = est_total_dur / est_total_sac / Fs
    # get average amplitude
    est_v_smooth = algorithms.v_denoise(original, Fs)
    est_total_amp = np.abs(est_v_smooth[est_detect_array == 1]).sum()
    # comment: divide Fs doesn't make sense, but lower error
    est_amp = est_total_amp / est_total_sac / Fs
    # calculate sigma using equation from the paper
    if Fs <= 500:
        alpha = 0.016 * Fs * est_sigma
        beta = 0.008 * Fs * np.sqrt(est_amp) * np.exp(5 * est_dur)
    else:
        alpha = (0.0032*Fs + 6.4) * est_sigma
        beta = (0.0016*Fs + 3.2) * np.sqrt(est_amp) * np.exp(5 * est_dur)

    # denoise signal
    denoised_signal = algorithms.cgtv(original, alpha, beta, ITER_N)

    # run VT algorithm on smooth out signal
    detection_array, total_sacs = algorithms.VT(
        denoised_signal, Fs, V_TH, DUR_TH, FIX_TH, MOVAVG)

    return denoised_signal, detection_array, total_sacs


def sacc_start_end(detection_array):
    """
    get the start and end of each saccade

    @param detection_array: detection array from process_data()

    @return: start (1) and end (-1) of each saccade
    """

    # shift detection array right by 1 unit
    detection_array2 = np.hstack((detection_array[0], detection_array[:-1]))
    result = detection_array - detection_array2
    return result
