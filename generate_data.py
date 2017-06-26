import numpy as np
import wave
import scipy.io.wavfile
import os
import shutil
from multiprocessing import Process

def load_file(filename):
    return wave.open(filename)


def rename():
    counter = 1
    for f in sorted(os.listdir("../clean/")):
        shutil.copy("../clean/"+f, "../clean/words_{}.wav".format(counter))
        counter += 1


def mix_and_match(mult):
    noisy_dir = "../noisy/"
    clean_dir = "../clean/"
    out_dir = "../mixed/"
    for i in range(1,1000):
        clean_index = np.random.randint(1,333)
        noisy_index = np.random.randint(1,9)

        try:
            clean_file = "words_{}.wav".format(clean_index)
            noisy_file = "ambient_{}.wav".format(noisy_index)
            out_file = "mixed_{}_from_{}.wav".format(i*mult, clean_index)
            mix_singles(noisy_dir + noisy_file, clean_dir + clean_file, out_dir + out_file)
            print i
        except Exception, e:
            print str(e)

def get_channels(li):
    return (li[:,0] + li[:,1])/2

def scale_sounds(data):
    """Scale sounds by a factor of the largest digit to avoid clipping"""
    mx = max(np.abs(data))
    b = 32767 / float(mx)
    data = data * b
    data = np.int16(data)
    return data

def normalize_volume(sound1, sound2):
    """normalizes ambient audio so as to avoid overpowering the clear speech"""
    a,b = np.max(np.abs(sound1)), np.max(np.abs(sound2))
    if a > b:
        pass
    elif b> a:
        k = b/float(a)
        if k < 1:
            k = 1        
        sound2 = sound2/k
    else:
        pass
    
    return np.array([int(x) for x in sound1],dtype=np.int16), np.array([int(x) for x in sound2], dtype=np.int16)



def mix_singles(noisy_file,clean_file,out_file):
    try:
        index = np.random.randint(1,51)
        noise = scipy.io.wavfile.read(noisy_file)[1]
        sr, clean = scipy.io.wavfile.read(clean_file)

        start_noise_pos = np.random.randint(0,noise.shape[0]-clean.shape[0])
        end_noise_pos = start_noise_pos + clean.shape[0]


        chunk = noise[start_noise_pos:end_noise_pos]
        chunk = get_channels(chunk)
        
        clean, chunk = normalize_volume(clean, chunk)
        new = clean + chunk
        scipy.io.wavfile.write(out_file,sr,new)
        # print "mixed {} and {} to get {}".format(noisy_file, clean_file,out_file)
    except Exception, e:
        print noisy_file, clean_file, out_file
        print str(e)

def break_long_audio(file_dir, file_name):
    sr, data = scipy.io.wavfile.read(file_dir + file_name)
    li = []
    for i in range(0,len(data),len(data)/348):
        li.append(data[i:i+len(data)/348])

    counter = 2
    # print li
    for x in li:
        scipy.io.wavfile.write("../clean/words_{}.wav".format(counter),sr, x)
        counter += 1

def run_in_parallel(functions):
    proc = []
    for f in functions:
        p = Process(target=f[0], args=(f[1],))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


if __name__ == "__main__":
    # do stuff
    noisy_dir = "../noisy/"
    clean_dir = "../clean/"
    out_dir = "../mixed/"
    # mix_and_match(noisy_dir,clean_dir,out_dir)
    # break_long_audio(clean_dir, "words_1.wav")
    functions = []
    for i in range(1,11):
      functions.append([mix_and_match,i])
    run_in_parallel(functions)

    # mix_singles(noisy_dir + "ambient_7.wav", clean_dir + "words_15.wav", out_dir + "a.wav")
        