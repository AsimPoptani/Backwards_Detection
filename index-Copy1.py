#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pandas as pd
import math
from PIL import Image
import sys
import random


# In[2]:


tf.__version__


# In[3]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[4]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])
  except RuntimeError as e:
    print(e)


# In[5]:


from utils import movie_images_to_dict


# In[6]:


trainning_folder="trainning_images"
test_folder="test_images"


# In[7]:


number_of_trainning_videos=3
number_of_test_videos=1


# In[8]:


trainning_data_set=[]
test_data_set=[]
# Loop through trainning images
for index in range(number_of_trainning_videos):
    trainning_data_set.append(pd.DataFrame(movie_images_to_dict.get_images_to_dict(f'{trainning_folder}/{index}/')).to_numpy())
# Loop through test images
for index in range(number_of_test_videos):
    test_data_set.append(pd.DataFrame(movie_images_to_dict.get_images_to_dict(f'test_images/{index}/')).to_numpy())


# In[9]:


pd.DataFrame(trainning_data_set[0])


# In[10]:


# Split in to groups
def split_in_groups(dataset,number_of_images=2,step=1):
#     Generate in groups
    dataset_gen=[dataset[i : i + number_of_images] for i in range(0, len(dataset), step) if i+number_of_images<len(dataset)]
#     Return as a numpy group
    return np.array(dataset_gen)


# Trainning groups
trainning_groups=[]
for i in range(number_of_trainning_videos):
    trainning_groups.append(split_in_groups(trainning_data_set[i],6))
trainning_groups=np.concatenate(trainning_groups,axis=0)                            
                            

# Test groups
test_groups=[]
for i in range(number_of_test_videos):
    test_groups.append(split_in_groups(test_data_set[i],6))                
test_groups=np.concatenate(test_groups,axis=0)


# In[11]:


input_shape=(int(800/8),int(1920/8),3)
input_shape


# In[12]:


resize_and_rescale = tf.keras.Sequential([
  K.layers.experimental.preprocessing.Resizing(input_shape[0],input_shape[1]),
  K.layers.experimental.preprocessing.Rescaling(1./255)
])
def resize_image(images,y):
    new_images=[]
    for image in images:
        new_images.append(resize_and_rescale(image))
    print(new_images)
    return tuple(new_images),y
    
# Gen
def trainning_group_gen():
    for trainning_group in trainning_groups:
        images=[]
        for record in trainning_group:
            image=Image.open(record[1]+record[0])
            image=image.resize((input_shape[1],input_shape[0]))
            images.append(np.array(image)/255)

        yield tuple(images),1
        yield tuple(images[::-1]),0
        
def trainning_group_gen2():
    for trainning_group in trainning_groups:
        images=[]
        for record in trainning_group:
            image=Image.open(record[1]+record[0])
            images.append(np.array(image))

        yield tuple(images),1
        yield tuple(images[::-1]),0
        
def test_group_gen():
    for test_group in test_groups:
        images=[]
        for record in test_group:
            image=Image.open(record[1]+record[0])
            image=image.resize((input_shape[1],input_shape[0]))
            images.append(np.array(image)/255)
        yield tuple(images),1
        yield tuple(images[::-1]),0


# In[13]:


tensor_trainning_dataset=tf.data.Dataset.from_generator(
    trainning_group_gen,
    output_signature=(
    (tf.TensorSpec(shape=input_shape, dtype=tf.float32),tf.TensorSpec(shape=input_shape, dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int8)
    )
).batch(20).prefetch(tf.data.AUTOTUNE)


tensor_trainning_dataset2=tf.data.Dataset.from_generator(
    trainning_group_gen2,
    output_signature=(
    (tf.TensorSpec(shape=(800,1920,3), dtype=tf.float32),tf.TensorSpec(shape=(800,1920,3), dtype=tf.float32),tf.TensorSpec(shape=(800,1920,3), dtype=tf.float32),tf.TensorSpec(shape=(800,1920,3), dtype=tf.float32),tf.TensorSpec(shape=(800,1920,3), dtype=tf.float32),tf.TensorSpec(shape=(800,1920,3), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int8)
    )
).batch(20).map(resize_image)


tensor_test_dataset=tf.data.Dataset.from_generator(
    test_group_gen,
    output_signature=(
    (tf.TensorSpec(shape=input_shape, dtype=tf.float32),tf.TensorSpec(shape=input_shape, dtype=tf.float32),tf.TensorSpec(shape=input_shape, dtype=tf.float32),tf.TensorSpec(shape=input_shape, dtype=tf.float32),tf.TensorSpec(shape=input_shape, dtype=tf.float32),tf.TensorSpec(shape=input_shape, dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int8)
    )
).batch(3).prefetch(tf.data.AUTOTUNE)


# In[14]:



tf.math.reduce_mean(np.array([[1,2,3],[2,3,4]],dtype=np.float32),axis=0)


# In[15]:



def block(block,num_filters):
    block=K.layers.Conv2D(num_filters,3,padding='same')(block)
    block=K.layers.LeakyReLU()(block)
    block=K.layers.BatchNormalization()(block)
    block=K.layers.Dropout(.1)(block)
    block=K.layers.MaxPool2D((3,3))(block)
    return block
start=K.layers.Input(shape=input_shape)
block_1=block(start,64)
block_2=block(block_1,128)
block_3=block(block_2,256)
block_4=block(block_3,512)
global_max=K.layers.GlobalMaxPool2D()(block_4)
conv=K.Model(inputs=start,outputs=global_max)


image_1=K.Input(input_shape)
image_2=K.Input(input_shape)
image_3=K.Input(input_shape)
image_4=K.Input(input_shape)
image_5=K.Input(input_shape)
image_6=K.Input(input_shape)

image_1_nn=conv(image_1)
image_2_nn=conv(image_2)
image_3_nn=conv(image_3)
image_4_nn=conv(image_4)
image_5_nn=conv(image_5)
image_6_nn=conv(image_6)

def add(images):
    return images[0]+images[1]+images[2]+images[3]+images[4]+images[5]

lambda_layer=K.layers.Lambda(add)([image_1_nn,image_2_nn,image_3_nn,image_4_nn,image_5_nn,image_6_nn])
fc_1=K.layers.Dense(200)(lambda_layer)
fc_1=K.layers.LeakyReLU()(fc_1)
fc_1=K.layers.Dense(200)(fc_1)
fc_1=K.layers.LeakyReLU()(fc_1)
fc_2=K.layers.Dense(1)(fc_1)
bce = tf.keras.losses.BinaryCrossentropy()
model=K.Model(inputs=[image_1,image_2,image_3,image_4,image_5,image_6],outputs=fc_2)


# In[16]:


K.utils.plot_model(model,show_shapes=True,expand_nested=True,show_dtype=True)


# In[17]:


# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
model.compile(
#     options = run_opts,
    loss=K.losses.BinaryCrossentropy(),
    optimizer=K.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)


# In[ ]:


history=model.fit(tensor_trainning_dataset2,epochs=5)
model.save('model')
