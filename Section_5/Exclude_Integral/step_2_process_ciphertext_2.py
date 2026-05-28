
import numpy as np
import time 
import os
import glob



def convert_txt_npy(file_path):

    save_name = file_path[:-3] + 'npy'

    start_time = time.time()

    with open(file_path, 'r') as file:
        data = file.read()  

    data_list = data.split()

    data_array = np.array(data_list, dtype=int)

    data_array = data_array.reshape(pow(2, 18), 105, 64)
    
    np.save(save_name, data_array)

    end_time = time.time()

    execution_time = end_time - start_time



if __name__ == "__main__":


    directory = 'cipher_txt_npy/'

    txt_files = glob.glob(os.path.join(directory, '*.txt'))


    for txt_file in txt_files:

        npy_file = txt_file.replace('.txt', '.npy')
        
        if not os.path.exists(npy_file):
            convert_txt_npy(txt_file)