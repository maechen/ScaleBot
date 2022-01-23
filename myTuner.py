import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

freq_sample = 48000  # hz
window_size = 48000  # window size of the DFT in samples
window_step = 12000  # step size
max_hps = 10  # max number of harmonic product spectrums
concert_pitch = 440

window_length = window_size / freq_sample  # window length in secs
sample_length = 1 / freq_sample  # length between 2 samples in secs
df = freq_sample / window_size  # frequency step width of the interpolated DFT

chromatic = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]


def return_pitch_note(actual_pitch):
    i = int(np.round(np.log2(actual_pitch / concert_pitch) * 12))
    pitch = concert_pitch * 2 ** (i / 12)
    note = chromatic[i % 12]
    return pitch, note


hanning_window = np.hanning(window_size)


def callback(data, frames, time, status):
    if not hasattr(callback, "samples"):
        callback.samples = [0 for _ in range(window_size)]
    if not hasattr(callback, "buffer"):
        callback.buffer = ["1", "2"]

    if status:
        print(status)
        return
    if any(data):
        callback.samples = np.concatenate((callback.samples, data[:, 0]))  # append samples
        callback.samples = callback.samples[len(data[:, 0]):]  # remove samples

        hanning_samps = callback.samples * hanning_window  # multiplying signal by hanning window
        mag_spec = abs(scipy.fftpack.fft(hanning_samps)[:len(hanning_samps) // 2])

        for i in range(int(27 / df)):
            mag_spec[i] = 0  # everything below 27 hz set to 0

        # interpolating spectrum
        interp_mag_spec = np.interp(np.arange(0, len(mag_spec), 1 / max_hps), np.arange(0, len(mag_spec)), mag_spec)
        interp_mag_spec = interp_mag_spec / np.linalg.norm(interp_mag_spec, ord=2, axis=0)

        # calculating harmonic product spectrum
        hps_spec = copy.deepcopy(interp_mag_spec)
        for i in range(max_hps):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(interp_mag_spec) / (i + 1)))],
                                       interp_mag_spec[::(i + 1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (freq_sample / window_size) / max_hps

        closest_pitch, closest_note = return_pitch_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        callback.buffer.insert(0, closest_note)  # ringbuffer
        callback.buffer.pop()

        # code was taken from not-chciken on GitHub
        os.system('cls' if os.name == 'nt' else 'clear')
        if callback.buffer.count(callback.buffer[0]) == len(callback.buffer):
            print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
        else:
            print(f"Closest note: ...")

    else:
        print('no input')


try:
    print("Finding pitch...")
    with sd.InputStream(channels=1, callback=callback, blocksize=window_step, samplerate=freq_sample):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))
