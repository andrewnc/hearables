import scipy.stats
import scipy.io.wavfile
import os
import numpy as np



def measure_SNR(filename):
    sr, data = scipy.io.wavfile.read(filename)
    return scipy.stats.signaltonoise(data)



if __name__ == "__main__":
    li = []
    for f in os.listdir("../mixed/"):
        li.append(measure_SNR("../mixed/" + str(f)))

    print max(li), min(li), sum(li)/float(len(li))
