# -*- coding: utf-8 -*-
"""
# Data Processing
Exploring and understanding data files is the fundamental but important step of the project.
"""
"""
# Data Collection
Go to the [Account tab of your user profile](https://www.kaggle.com/me/account) and select Create API Token. To get kaggle.json, which contains your username and key.
"""

#Set the enviroment variables
import os
os.system("sudo pip install kaggle")
os.environ['KAGGLE_USERNAME'] = "" # your username here
os.environ['KAGGLE_KEY'] = "" # your key here
kaggle competitions download -c whale-categorization-playground #down load data here

import numpy as np
import pandas as pd

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

# Python 2/3 compatibility
from __future__ import print_function, division

import itertools
import time

import numpy as np
import matplotlib.pyplot as plt

# Colors from Colorbrewer Paired_12
colors = [[31, 120, 180], [51, 160, 44]]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

# functions to show an image
def imshow(img):
    """
    :param img: (PyTorch Tensor)
    """
    # unnormalize
    img = img / 2 + 0.5     
    # Convert tensor to numpy array
    npimg = img.numpy()
    # Color channel first -> color channel last
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_losses(train_history, val_history):
    x = np.arange(1, len(train_history) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_history, color=colors[0], label="Training loss", linewidth=2)
    plt.plot(x, val_history, color=colors[1], label="Validation loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training and validation loss")
    plt.show()

os.system("sudo pip install scipy")
os.system("sudo pip install Pillow")
os.system("sudo pip install imagehash")

"""### Duplication Detection 
Using the PIL and the imagehash library, we got their hash, shape, mode, number of each id, and whether the image belonged to the “new whale”. 

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
import imagehash

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

TRAIN_IMG_PATH = os.getcwd()+"/train/train"

def getImageMetaData(file_path):
    with Image.open(file_path) as img:
        img_hash = imagehash.phash(img)
        return img.size, img.mode, img_hash

def get_train_input():
    train_input = pd.read_csv(os.getcwd()+"/train.csv")
    
    m = train_input.Image.apply(lambda x: getImageMetaData(TRAIN_IMG_PATH + "/" + x))
    train_input["Hash"] = [str(i[2]) for i in m]
    train_input["Shape"] = [i[0] for i in m]
    train_input["Mode"] = [str(i[1]) for i in m]
    train_input["New_Whale"] = train_input.Id == "new_whale"
    
    
    img_counts = train_input.Id.value_counts().to_dict()
    train_input["Id_Count"] = train_input.Id.apply(lambda x: img_counts[x])
    return train_input

train_input = get_train_input()

train_input.head()

t = train_input.Hash.value_counts()
t = t.loc[t>1]
print("There are {} duplicate images.".format(np.sum(t)-len(t)))
t.head()

import collections

def plot_images(path, imgs):
    assert(isinstance(imgs, collections.Iterable))
    imgs_list = list(imgs)
    nrows = len(imgs_list)
    if (nrows % 2 != 0):
        nrows = nrows + 1 

    plt.figure(figsize=(18, 6*nrows/2))
    for i, img_file in enumerate(imgs_list):
        with Image.open(path + "/" + img_file) as img:
            ax = plt.subplot(nrows/2, 2, i+1)
            ax.set_title("#{}: '{}'".format(i+1, img_file))
            ax.imshow(img)
        
    plt.show()

print("Some examples:")
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[0]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[5]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[9]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[66]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[321]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[557]].Image)

"""#### Data Cleaning 
Determine duplicate images using the hash, and clean the duplicate images
"""

t = train_input.Hash.value_counts()
t = t[t > 1]
duplicates_df = pd.DataFrame(t)

# get the Ids of the duplicate images
duplicates_df["Ids"] =list(map(
            lambda x: set(train_input.Id[train_input.Hash==x].values), 
            t.index))
duplicates_df["Ids_count"] = duplicates_df.Ids.apply(lambda x: len(x))
duplicates_df["Ids_contain_new_whale"] = duplicates_df.Ids.apply(lambda x: "new_whale" in x)

duplicates_df.head()

import copy

train_input.drop_duplicates(["Hash", "Id"], inplace = True)

drop_hash = duplicates_df.loc[(duplicates_df.Ids_count>1) & (duplicates_df.Ids_contain_new_whale==True)].index
tb1 = copy.deepcopy(train_input[(train_input.Hash.isin(drop_hash) & (train_input.Id=="new_whale"))])
train_input.drop(train_input.index[(train_input.Hash.isin(drop_hash) & (train_input.Id=="new_whale"))], inplace=True)

drop_hash = duplicates_df.loc[(duplicates_df.Ids_count>1) & ((duplicates_df.Ids_count - duplicates_df.Ids_contain_new_whale)>1)].index
tb2 = copy.deepcopy(train_input[train_input.Hash.isin(drop_hash)])
train_input.drop(train_input.index[train_input.Hash.isin(drop_hash)], inplace=True)

assert(np.sum(train_input.Hash.value_counts()>1) == 0)

a1 = copy.deepcopy(train_input.values[:,0])
a2 = copy.deepcopy(train_input.values[:,1])
a3 = copy.deepcopy(train_input.values[:,2])
df = {'Image': a1, 'Id': a2, 'Hash':a3}
new_tb = pd.DataFrame(data=df)
print(new_tb)

a1 = copy.deepcopy(train_input.values[:,0])
a2 = copy.deepcopy(train_input.values[:,1])
df = {'Image': a1, 'Id': a2}
new_tb = pd.DataFrame(data=df)
new_tb

"""Cleaned training file generation. """

classes = np.unique(a2)
labels = []
classes = classes.tolist()
for i in a2:
  labels.append(classes.index(i))
new_tb['id'] = labels
new_tb.to_csv(os.getcwd()+"cleaned_train.csv", index=False)

"""#### File Generation 
Create the test file.
"""

test = pd.read_csv(os.getcwd()+"/sample_submission.csv", sep=",")
test['Id'] = '?'
test['id'] = 0
test

test.to_csv(os.getcwd()+"cleaned_test.csv", index=False)

"""### Class Distribution 
View the number of images in each class
"""

data = pd.read_csv(os.getcwd()+"/cleaned_train.csv", sep=",")
print(data)

num = data.Id.value_counts()
num

plt.title('Distribution of classes excluding new_whale')
data.Id.value_counts()[1:].plot(kind='hist', logy=True, legend=True);

from collections import Counter

distribution = Counter(data.Id.value_counts().values)
plt.bar(range(len(distribution)), list(distribution.values())[::-1], align='center')
plt.xticks(range(len(distribution)), list(distribution.keys())[::-1])
plt.title("Distribution of classes")

plt.show()

distribution[1]

"""#### Oversampling 
Identify the classes that have number of images less than ten.
Will increase them to ten in the data augmentation part.
"""

c1 = list(num[num == 1].index)

data[data['Id'].isin(c1)]

for i in range(1,10):
  temp_list = list(num[num == i].index)
  temp_df = data[data['Id'].isin(temp_list)]
  temp_df.to_csv(os.getcwd()+f"train{i}.csv", index=False)