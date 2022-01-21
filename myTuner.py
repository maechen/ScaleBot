import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

freqSample = 48000 #hz
windowSize = 48000 # window size of the DFT in samples
windowStep = 12000 # step size
hps = 10 # max number of harmonic product spectrums
concertPitch = 440

windowLength = windowSize / freqSample # window length in secs
sampleLength = 1 / freqSample # length between 2 samples in secs
df = freqSample / windowSize # frequency step width of the interpolated DFT

chromatic = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]
def returnPitchNote(actualPitch):
    i = int(np.round(np.log2(actualPitch / concertPitch) * 12))
    pitch = concertPitch * 2 ** (i / 12)
    note = chromatic[i % 12]
    return pitch, note

hannWindow = np.hanning(windowSize)
def callback(data, frames, time, status):

  if not hasattr(callback, "samples"):
    callback.samples = [0 for _ in range(windowSize)]
  if not hasattr(callback, "buffer"):
    callback.buffer = ["1","2"]

  if status:
    print(status)
    return
  if any(data):
    callback.samples = np.concatenate((callback.samples, data[:, 0])) # append samples
    callback.samples = callback.samples[len(data[:, 0]):] # remove samples

    hannSamps = callback.samples * hannWindow # multiplying signal by hann window
    magSpec = abs(scipy.fftpack.fft(hannSamps)[:len(hannSamps)//2])

    for i in range(int(27/df)):
      magSpec[i] = 0 # everything below 27 hz set to 0

    # interpolating spectrum
    interpMagSpec = np.interp(np.arange(0, len(magSpec), 1/hps), np.arange(0, len(magSpec)), magSpec)
    interpMagSpec = interpMagSpec / np.linalg.norm(interpMagSpec, ord=2, axis=0)

    # calculating harmonic product spectrum
    hpsSpec = copy.deepcopy(interpMagSpec)
    for i in range(hps):
      tmp_hps_spec = np.multiply(hpsSpec[:int(np.ceil(len(interpMagSpec)/(i+1)))], interpMagSpec[::(i+1)])
      if not any(tmp_hps_spec):
        break
      hpsSpec = tmp_hps_spec

    max_ind = np.argmax(hpsSpec)
    max_freq = max_ind * (freqSample/windowSize) / hps

    closest_pitch, closest_note = returnPitchNote(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    callback.buffer.insert(0, closest_note) # ringbuffer
    callback.buffer.pop()

    # code was taken from not-chciken on GitHub
    os.system('cls' if os.name=='nt' else 'clear')
    if callback.buffer.count(callback.buffer[0]) == len(callback.buffer):
      print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
    else:
      print(f"Closest note: ...")

  else:
    print('no input')

try:
  print("Starting HPS guitar tuner...")
  with sd.InputStream(channels=1, callback=callback, blocksize=windowStep, samplerate=freqSample):
    while True:
      time.sleep(0.5)
except Exception as exc:
  print(str(exc))