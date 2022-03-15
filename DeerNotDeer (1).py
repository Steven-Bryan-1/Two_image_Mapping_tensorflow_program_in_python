#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from tqdm import tqdm

#below is my path for the files to train.  All the files can be changed to compare 
#any two things as you would like
DATADIR = "C:Desktop\DeerProject"

CATEGORIES = ["Deer", "NotDeer"]

for category in CATEGORIES:  # deer and notdeer
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!


# In[2]:


print(img_array)


# In[3]:


print(img_array.shape)


# In[4]:


IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[5]:


new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[6]:


training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))


# In[7]:


import random

random.shuffle(training_data)


# In[8]:


for sample in training_data[:50]:
    print(sample[1])


# In[9]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:





# In[10]:


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[11]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[12]:




import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle


X = X/255.0
y=np.array(y)  #need to make y a numpy array
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=34, epochs=3, validation_split=0.3)


# In[13]:


from tensorflow.keras.callbacks import TensorBoard


# In[14]:


NAME = "Deer-NotDeer-CNN"

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


# In[15]:


model.fit(X, y,
          batch_size=32,
          epochs=3,
          validation_split=0.3,
          callbacks=[tensorboard])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



          


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
y=np.array(y) 
X = X/255.0

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])

            model.save('64x3-DeerNotDeer.model')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


import os
from PIL import Image

img_dir = "C:Desktop\notdeer.jpg"
for filename in os.listdir(img_dir):
    try :
        with Image.open(img_dir + "/" + filename) as im:
             print('ok')
    except :
        print(img_dir + "/" + filename)
        os.remove(img_dir + "/" + filename)


# In[ ]:





# In[53]:


import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

CATEGORIES = ["Deer", "NotDeer"]


def plot_value_array(i, predictions_array, true_label):
  true_label = true_label
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  
def prepare(filename):
   
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-DeerNotDeer.model")
#test=prepare('C:Desktop\Rabbittest.jpeg')  #create a variable with the pathname  MAKE SURE YOU HAVE THE FILE TYPE JPG OR JPEG!!!!

deer1prep= prepare('C:Desktop\deer1.jpg')




#rabbitorSquirrel = model.predict(test)  #does the prediction of the variable "test" anove 
deer1 = model.predict(deer1prep)


#rabbitorSquirrel2 = model.predict(test2)  #does the prediction of the variable "test" above 


i=0
test_label=0   #0 for circle, 1 for squirrell
test_label2=1

#predictions = model.eval(feed_dict = {x:circleorStest1prep})
notdeer1prep=prepare('C:Desktop\girl.jpg')  #create a variable with the pathname  MAKE SURE YOU HAVE THE FILE TYPE JPG OR JPEG!!!!
notdeer1 = model.predict(notdeer1preprep)

print(deer1)  # will be a list in a list.
print(CATEGORIES[int(deer1[0][0])])
plt.figure(figsize=(6,3))
plt.subplot(1,2,2)
plot_value_array(i, deer1[i],  test_label)
plt.show()
#plt.imshow(new_array, cmap='gray')
#plt.show()


#probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
#predictions = probability_model.predict(circleorStest2prep)
#print(predictions)
#np.argmax(predictions[0])


print(notdeer1)  # will be a list in a list.
print(CATEGORIES[int(notdeer1[0][0])])
plt.figure(figsize=(6,3))
plt.subplot(1,2,2)
plot_value_array(i, notdeer1[i],  test_label2)
plt.show()
#plt.imshow(new_array2, cmap='gray')
#plt.show()


#for i, logits in enumerate(circleorSquirrel3):
#  class_idx = tf.argmax(logits).numpy()
#  p = tf.nn.softmax(logits)[class_idx]
#  name = CATEGORIES[class_idx]
#  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

    
#probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])








# In[ ]:





# In[ ]:




