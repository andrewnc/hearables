import scipy.io.wavfile
import numpy as np
import os

def get_data(n=0):
    clean_dir = "../clean/"
    mixed_dir = "../mixed/"
    mixed_files = list(os.listdir(mixed_dir))
    #choose the number of files to get
    if n == 0:
        #TODO: get all files
        input_data = []
        output_data = []

        input_data_names = []
        output_data_names = []

        for file in mixed_files:
            clean_indx = file.split("_")[3].split(".")[0]
            input_data_names.append(file)
            output_data_names.append("nwords_{}.wav".format(clean_indx))

        for name in input_data_names:
            input_data.append(scipy.io.wavfile.read(mixed_dir + name)[1])

        for name in output_data_names:
            output_data.append(scipy.io.wavfile.read(clean_dir + name)[1])

        return input_data,output_data

            
    else:
        #initialize data and names of data lists
        input_data = []
        output_data = []

        input_data_names = []
        output_data_names = []

        # for the number of specified files
        for i in range(1,n+1):
            #initialize mixed files list and clean files list, both temporary holds
            m_files = []

            #choose a random base clean file
            clean_indx = np.random.randint(1,332)

            #find all of the mixed files that were created from that clean file
            for file in mixed_files:
                if "from_{}.wav".format(clean_indx) in file:
                    #keep track of the clean file so that it can be used as ground truth.
                    m_files.append(file)

            #choose a random mixed file from the list of all mixed files associated with the base file
            mixed_indx = np.random.randint(0,len(m_files))
            input_data_names.append(m_files[mixed_indx])
            output_data_names.append("nwords_{}.wav".format(clean_indx))

        #get the timeseries data for those files
        for name in input_data_names:
            input_data.append(scipy.io.wavfile.read(mixed_dir + name)[1])

        for name in output_data_names:
            output_data.append(scipy.io.wavfile.read(clean_dir + name)[1])


        return input_data, output_data


if __name__ == "__main__":
    input, output = get_data( n = 50 )