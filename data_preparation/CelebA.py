# author: Faezeh Mosayyebi
# Spliting CelebA dataset based on the config file.

import shutil
import os
import pandas as pd
from utils import *

def celeba_preparation(data_dir, config_dir, output_dir) -> None:

    path_valiadtor(data_dir)
    directory_maker(output_dir)

    if not os.path.isdir(os.path.join(output_dir, 'male')):
        os.mkdir(os.path.join(output_dir, 'male'))

    if not os.path.isdir(os.path.join(output_dir, 'female')):
        os.mkdir(os.path.join(output_dir, 'female'))

    config = pd.read_csv(config_dir).values

    for i in range(len(config)):

        if config[i][1] == 1: # gender information 1 = male, -1 = female     
            image = data_dir +'/'+ config[i][0]
            shutil.move(image, os.path.join(output_dir, 'male'))

        elif config[i][1] == -1:   
            image = data_dir +'/'+ config[i][0]
            shutil.move(image, os.path.join(output_dir, 'female'))

if __name__ == "__main__":

    celeba_preparation(data_dir, config_dir, output_dir)