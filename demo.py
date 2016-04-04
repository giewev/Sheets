from sheets.learning import SoundClassifier
import numpy as np
from scipy.io import wavfile

note_model = SoundClassifier('notes')
# for x in note_model.classes:
# 	print(x.ljust(3) + "  " + str(note_model.data_buckets[note_model.classes[x]].shape))

print(note_model.accuracy_ratio())
for epoch in range(1000):
	for example in range(1000):
		note_model.train_single()
	print(note_model.accuracy_ratio())