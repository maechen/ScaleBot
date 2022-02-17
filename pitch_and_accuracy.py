import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
from fastapi import BackgroundTasks, FastAPI

freq_sample = 48000  # hz
window_size = 48000  # window size of the DFT in samples
window_step = 12000  # step size
max_hps = 10  # max number of harmonic product spectrums
white_noise_thresh = .2
power_thresh = .000001
concert_pitch = 440

window_length = window_size / freq_sample  # window length in secs
sample_length = 1 / freq_sample  # length between 2 samples in secs
df = freq_sample / window_size  # frequency step width of the interpolated DFT
octave_bands = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

chromatic = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]
actual_notes = []
actual_pitches = []


def return_pitch_note(actual_pitch):
    i = int(np.round(np.log2(actual_pitch / concert_pitch) * 12))
    pitch = concert_pitch * 2 ** (i / 12)
    note = chromatic[i % 12]
    return pitch, note


hanning_window = np.hanning(window_size)


def callback(data, frames, time, status):
    global actual_notes, actual_pitches

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
        mag_spec = abs(scipy.fftpack.fft(hanning_samps)[:len(hanning_samps) // 2])  # only look at last few seconds

        for i in range(int(250 / df)):
            mag_spec[i] = 0  # everything below 250 hz set to 0

        # suppresses everything below average energy per frequency
        for j in range(len(octave_bands) - 1):
            start = int(octave_bands[j] / df)
            end = int(octave_bands[j + 1] / df)
            end = end if len(mag_spec) > end else len(mag_spec)
            avg_energy = (np.linalg.norm(mag_spec[start:end], ord=2, axis=0) ** 2) / (end - start)
            avg_energy = avg_energy ** 0.5
            for i in range(start, end):
                mag_spec[i] = mag_spec[i] if mag_spec[i] > white_noise_thresh * avg_energy else 0

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
        if callback.buffer.count(callback.buffer[0]) == len(callback.buffer):
            print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
            actual_notes.append(closest_note)
            actual_pitches.append(closest_pitch)
        else:
            print(f"Closest note: ...")
    else:
        print('no input')


is_recording = False
recording_completed = False

app = FastAPI()


def start_recording():
    global is_recording, recording_completed, actual_notes, actual_pitches, matched_scale

    actual_notes = []
    actual_pitches = []
    matched_scale = ""
    is_recording = True
    recording_completed = False
    try:
        print("Starting HPS guitar tuner...")
        sd.InputStream(channels=1, callback=callback, blocksize=window_step, samplerate=freq_sample)
        with sd.InputStream(channels=1, callback=callback, blocksize=window_step, samplerate=freq_sample):
            while is_recording:
                time.sleep(0.5)
        recording_completed = True
    except Exception as exc:
        print(str(exc))


def stop_recording():
    global is_recording, recording_completed, actual_notes, actual_pitches

    if is_recording:
        print("Stopping HPS guitar tuner...")
        is_recording = False
        while not recording_completed:
            time.sleep(0.1)


def recording_status():
    global is_recording, recording_completed

    return {
        "recording": {
            "is_recording": is_recording,
            "recording_completed": recording_completed,
        }
    }


d_major_notes = ["D", "E", "F#", "G", "A", "B", "C#", "D"]
d_major_pitches = [293.66, 329.622, 369.988, 391.989, 440, 493.883, 554.365, 622.254]
c_major_notes = ["C", "D", "E", "F", "G", "A", "B", "C"]
c_major_pitches = [261.629, 293.668, 329.631, 349.232, 392, 440.005, 493.889, 523.257]
matched_scale = ""
individual_percent = []
percent_accuracy = 0


@app.get("/start")
def api_start(background_tasks: BackgroundTasks):
    global is_recording
    if is_recording:
        return "recording already started"
    background_tasks.add_task(start_recording)
    return recording_status()


@app.get("/stop")
def api_stop():
    global actual_notes, actual_pitches, d_major_notes, d_major_pitches, c_major_notes, c_major_pitches, matched_scale,\
        individual_percent, percent_accuracy
    if recording_completed:
        return "recording already stopped"
    stop_recording()

    actual_notes = actual_notes[1:]
    actual_pitches = actual_pitches[1:]
    actual_pitches = list(dict.fromkeys(actual_pitches))
    d_scale = all(item in actual_notes for item in d_major_notes)
    c_scale = all(item in actual_notes for item in c_major_notes)
    individual_percent = []

    # identifies what scale
    if d_scale is True:
        matched_scale = "d_major"
    elif c_scale is True:
        matched_scale = "c_major"
    else:
        matched_scale = "doesn't match"

    if matched_scale == "d_major":
        for i in range(0, len(actual_pitches)): 
            individual_percent.append(actual_pitches[i] / d_major_pitches[i])
        percent_accuracy = round((sum(individual_percent) / len(individual_percent)) * 100, 2)
    elif matched_scale == "c_major":
        for i in range(0, len(actual_pitches)):  
            individual_percent.append(actual_pitches[i] / c_major_pitches[i])
        percent_accuracy = round((sum(individual_percent) / len(individual_percent)) * 100, 2)

    if not individual_percent == []:
        return {
            "scale": matched_scale,
            "percent_accuracy": str(percent_accuracy) + "%",
        }
    else:
        return {
            "scale": matched_scale,
            "percent_accuracy": matched_scale,
        }


@app.get("/status")
def api_status():
    global actual_notes, actual_pitches, matched_scale, individual_percent, percent_accuracy

    if not individual_percent == []:
        return {
            "recording_status": recording_status(),
            "actual_notes": actual_notes,
            "actual_pitches": actual_pitches,
            "scale": matched_scale,
            "each_note_accuracy": individual_percent,
            "percent_accuracy": str(percent_accuracy) + "%",
        }
    else:
        return {
            "recording_status": recording_status(),
            "actual_notes": actual_notes,
            "actual_pitches": actual_pitches,
            "scale": matched_scale,
            "each_note_accuracy": matched_scale,
            "percent_accuracy": matched_scale,
        }
