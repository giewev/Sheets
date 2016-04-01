from sheets.learning import SoundClassifier
import numpy as np
from scipy.io import wavfile

fs, data = wavfile.read('notes/a3/a3-1.wav')

note_model = SoundClassifier('notes')
for x in note_model.data_buckets:
	print(x.shape)