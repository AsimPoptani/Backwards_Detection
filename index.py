# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from PIL import Image
import random

# tf.config.threading.set_intra_op_parallelism_threads(3)
# tf.config.threading.set_inter_op_parallelism_threads(3)
# %%
from utils import movie_images_to_dict


# %%
trainning_folder="trainning_images"
test_folder=""


# %%
trainning_data_set=[]
test_data_set=[]
# Loop through trainning images
for index in range(3):
    trainning_data_set.append(pd.DataFrame(movie_images_to_dict.get_images_to_dict(f'/media/Data/backwards_detection/trainning_images/{index}/')).to_numpy())
# Loop through test images
for index in range(1):
    test_data_set.append(pd.DataFrame(movie_images_to_dict.get_images_to_dict(f'/media/Data/backwards_detection/test_images/{index}/')).to_numpy())


# %%
pd.DataFrame(test_data_set[0])


# %%
from typing import List, Tuple, Union
def shape(ndarray: Union[List, float]) -> Tuple[int, ...]:
    if isinstance(ndarray, list):
        # More dimensions, so make a recursive call
        outermost_size = len(ndarray)
        row_shape = shape(ndarray[0])
        return (outermost_size, *row_shape)
    else:
        # No more dimensions, so we're done
        return ()


# %%
# conversion methods 
# will generate a array of x frames
def every_x_frame(data,x=5):
    size=data.shape[0]
    left_over=size%x
    n_times=math.floor(size/x)
    data=data[:-left_over,:]
    split=np.split(data,n_times)
    return split

def negative_frames(data,x):
    data_to_be_shuffled=data.copy()
    
    np.random.shuffle(data_to_be_shuffled)
    random=every_x_frame(data_to_be_shuffled,x)
    flip=every_x_frame(np.flip(data,axis=0),x)
    return flip+random

# generates positive and negative tests for x frames
def gen_every_x_frames(data,x=5):
    data=data.copy()
    positive=every_x_frame(data,x)
    negative=negative_frames(data,x)
    return [positive,negative]

# generates positive and negative tests for x frames
def gen_every_x_frames_skip(data,step=2,x=5):
    data=data.copy()
    data=data[::step]
    return gen_every_x_frames(data,x)


# %%
NUMBER_OF_IMAGES=5
def trainning_data_set_gen():
    data_set_pos=[]
    data_set_neg=[]
    for trainning in trainning_data_set:
        # generate cases for NUMBER_OF_IMAGES images and a skip of 2
        positive, negative=gen_every_x_frames_skip(trainning,x=NUMBER_OF_IMAGES)
        data_set_pos=data_set_pos+positive
        data_set_neg=data_set_neg+negative
    return data_set_pos,data_set_neg
def test_data_set_gen():
    data_set_pos=[]
    data_set_neg=[]
    for test in test_data_set:
        # generate cases for 5 images and a skip of 2
        positive, negative=gen_every_x_frames_skip(test, x=NUMBER_OF_IMAGES)
        data_set_pos=data_set_pos+positive
        data_set_neg=data_set_neg+negative
    return data_set_pos,data_set_neg

def pick_x_amount(data,x):
    return random.sample(data,x)

def open_images_prep_keras(posData, negData, limit, random=True):
    # take positive and negative data and limit
    if random:
        posData=pick_x_amount(posData,limit)
        negData=pick_x_amount(negData,limit)
    else:
        posData=posData[0:limit]
        negData=negData[0:limit]
    # image_paths Y
    data_set=[]
    for pos_images in posData:
        image_paths=[]
        for image in pos_images:
            image_paths.append(image[1]+image[0])
        xy=[image_paths,1]
        data_set.append(xy)
    for neg_images in negData:
        image_paths=[]
        for image in neg_images:
            image_paths.append(image[1]+image[0])
        xy=[image_paths,0]
        data_set.append(xy)
    data_set=np.array(data_set)
    np.random.shuffle(data_set)
    for data in data_set:
        image_data={}
        for i,image_path in enumerate(data[0]):
            image_data[f"input_{i}"]=np.array(Image.open(image_path),dtype=np.float32)/255
        yield (image_data,np.array(data[1],dtype=np.int8))

def trainning_images_gen():
    keras_trainning_data_set_pos, keras_trainning_data_set_neg= trainning_data_set_gen()
    return open_images_prep_keras(keras_trainning_data_set_pos,keras_trainning_data_set_neg,10000)
trainning_images_gen()

def test_images_gen():
    keras_test_data_set_pos, keras_test_data_set_neg= test_data_set_gen()
    return open_images_prep_keras(keras_test_data_set_pos,keras_test_data_set_neg,1000)


# %%



# %%

x_gen_inputs={}
x_gen_inputs_shapes={}
for i in range(NUMBER_OF_IMAGES):
    x_gen_inputs[f"input_{i}"]=tf.float32
    x_gen_inputs_shapes[f"input_{i}"]=tf.TensorShape([800,1920,3])
xy_gen_inputs_output=(x_gen_inputs,tf.int8)
xy_gen_inputs_output_shape=(x_gen_inputs_shapes,tf.TensorShape([]))
trainning_images_gen_tensor=tf.data.Dataset.from_generator(
    trainning_images_gen,
    xy_gen_inputs_output,
    xy_gen_inputs_output_shape
    )

test_images_gen_tensor=tf.data.Dataset.from_generator(
    test_images_gen,
    xy_gen_inputs_output,
    xy_gen_inputs_output_shape
    )
training_dataset_with_batch_and_prefetch=trainning_images_gen_tensor.batch(2).prefetch(2)
test_dataset_with_batch_and_prefetch=test_images_gen_tensor.batch(2).prefetch(2)


# %%
# convert input to output
def start_of_cnn(i):
    image_input=tf.keras.layers.Input(shape=(800,1920,3),name=f"input_{i}")
    x=tf.keras.layers.Conv2D(filters=10,kernel_size=[2,2],activation='relu',padding='valid')(image_input)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Dropout(.1)(x)
    x=tf.keras.layers.Conv2D(filters=20,kernel_size=[2,2],activation='relu',padding='valid')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Dropout(.1)(x)
    x=tf.keras.layers.Conv2D(filters=40,kernel_size=[2,2],activation='relu',padding='valid')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Dropout(.1)(x)
    x=tf.keras.layers.Conv2D(filters=80,kernel_size=[2,2],activation='relu',padding='valid')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Dropout(.1)(x)
    x=tf.keras.layers.Conv2D(filters=160,kernel_size=[2,2],activation='relu',padding='valid')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Dropout(.1)(x)

    x=tf.keras.layers.Flatten()(x)
    # x=tf.keras.Model(inputs=image_input,outputs=x) #(None, 48, 118, 25) 
    return image_input,x
inputs=[]
combined_layers=[]
for i in range(NUMBER_OF_IMAGES):
    input_cnn, combined_layer = start_of_cnn(i)
    inputs.append(input_cnn)
    combined_layers.append(combined_layer)

combined=tf.keras.layers.concatenate(axis=1,inputs=combined_layers)
x=tf.keras.layers.Dense(200, activation="relu")(combined)
x=tf.keras.layers.Dense(200, activation="relu")(x)
x=tf.keras.layers.Dense(1, activation="softmax")(x)
x=tf.keras.Model(inputs=inputs,outputs=x)
x.summary()


# %%
x.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x.fit(training_dataset_with_batch_and_prefetch,validation_data=test_dataset_with_batch_and_prefetch, epochs=100)


