
# coding: utf-8

# In[7]:

from PIL import Image
from fnmatch import fnmatch

import numpy as np
from eval_segm import *
import os


def read_unlabeled_image_list(image_list_file):

    root = image_list_file
    pattern = "*.png"
    images = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                images.append(path +"/"+ name)

    return images



# In[8]:

image_list_g = read_unlabeled_image_list("gt/")
image_list_p = read_unlabeled_image_list("pre/")


# In[30]:

pixel_accuracy_ = []
mean_accuracy_ = []
mean_IU_ = []
frequency_weighted_IU_ = []
for i in range(len(image_list_g)):
    g = np.array(Image.open(image_list_g[i]))
    p = np.array(Image.open(image_list_p[i]))
    
    pixel_accuracy_.append(pixel_accuracy(g,p))
    mean_accuracy_.append(mean_accuracy(g,p))
    mean_IU_.append(mean_IU(g,p))
    frequency_weighted_IU_.append(frequency_weighted_IU(g,p))
    
print("pixel_accuracy :        %",  round(np.mean(np.array(pixel_accuracy_       )),2))
print("mean_accuracy  :        %",  round(np.mean(np.array(mean_accuracy_        )),2))
print("mean_IU :               %",  round(np.mean(np.array(mean_IU_              )),2))
print("frequency_weighted_IU : %",  round(np.mean(np.array(frequency_weighted_IU_)),2))


# In[19]:

pixel_accuracy_(g,p)


# In[ ]:



