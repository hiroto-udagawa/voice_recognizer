import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 



audio = pd.read_csv('testaudio.csv')
audio = np.array(audio)
audio = np.delete(audio, 0,1)
final = np.array([])

def compress(audio, number):
	x = []
	clean = np.array(x)
	audio = audio[np.argmax(audio)-17000 :np.argmax(audio)+17000]
	
	#cuts the audio file in to 150 different strips and adds it together
	for i in range(0,len(audio)-(len(audio)/75),len(audio)/75):
		total = 0
		for x in range(len(audio)/75):
			total += abs(audio[i+x])
		clean = np.append(clean,total)

	#normalizes for the maximum value to be 1
	maximum = max(clean)
	for i in range(0,len(clean)):
		clean[i] = clean[i]/maximum
		
	clean = clean[np.argmax(clean)-35 :np.argmax(clean)+35]
	#plt.plot(clean)
	#plt.show()
	clean = np.insert(clean,0,number)
	global final
	final = np.concatenate([final,clean], axis=1)
	return clean

def reshaper():
	return np.reshape(final, (final.size/71, 71))
	

compress(audio,1)
compress(audio,2)
np.savetxt("foo.csv",reshaper(),delimiter=",")