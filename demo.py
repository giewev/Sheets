from sheets.learning import SoundClassifier
import numpy as np
from scipy.io import wavfile

note_model = SoundClassifier('notes', frame_jump = 200)
# for x in note_model.classes:
# 	print(x.ljust(3) + "  " + str(note_model.data_buckets[note_model.classes[x]].shape))

print("Done with loading data")
print(note_model.accuracy_ratio())
for epoch in range(1000):
	for example in range(1000):
		note_model.train(10)
	print(note_model.accuracy_ratio())