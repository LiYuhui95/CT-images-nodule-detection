# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:36:30 2017

@author: yuhui
"""
import os
import numpy as np
import cPickle as pickle
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import math
from PIL import Image

PATCH_SIZE = 40

def load_dataset(file_path = None):
    print 'start reading files....'
    dataset = []
    label = []
    if file_path == None:
        file_path = os.getcwd()
    for file_name in os.listdir(file_path):
        if file_name.endswith('0.pkl'):
            with open(file_name,'rb') as f:
                tmp_dataset = pickle.load(f)
                tmp_label = pickle.load(f)
            dataset.extend(tmp_dataset)
            label.extend(tmp_label)
    
    print 'reading files done'
    print len(label)
    dataset = np.array(dataset)
    label = np.array(label)
    p_index = []
    n_index = []
    for i in xrange(len(label)):
        if label[i] == 0:
            n_index.append(i)
        else:
            p_index.append(i)
    p_sample, p_label = dataset[p_index], label[p_index]
    n_sample, n_label = dataset[n_index], label[n_index]
    
    index = [i for i in range(len(n_label))] 
    np.random.shuffle(index)
    n_sample = n_sample[index]
    n_label = n_label[index]
    
    positive_number = len(p_label)
    n_sample = n_sample[:positive_number]
    n_label = n_label[:positive_number]
    
    dataset = np.concatenate((p_sample,n_sample), axis=0)
    label = np.concatenate((p_label,n_label), axis=0)
    dataset = dataset.reshape(-1, 1, PATCH_SIZE, PATCH_SIZE)
    index = [i for i in range(len(dataset))] 
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    
    print 'shuffle done'
    print len(label)
    
    with open('test_sample.pkl', 'wb') as f:
        pickle.dump(dataset, f)
        pickle.dump(label, f)
    return dataset, label

def get_val(train_x, train_y):
    valid_len = int(len(train_x) / 10)
    train_x = train_x / np.float32(255.0)
    train_x, val_x = train_x[:-valid_len], train_x[-valid_len:]
    train_y, val_y = train_y[:-valid_len], train_y[-valid_len:]
    
    return train_x, train_y, val_x, val_y
    
def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, 64, 64)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model

def vgg_like_model():
    model = Sequential()  
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, PATCH_SIZE, PATCH_SIZE)))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(32, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(64, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Flatten())  
    model.add(Dense(256))  
    model.add(Activation('relu'))  
    model.add(Dropout(0.5))  
      
    model.add(Dense(1))  
    model.add(Activation('sigmoid'))  
     
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model

def cnn_model_experiment_layer_7():
    model = Sequential()  
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, PATCH_SIZE, PATCH_SIZE)))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(32, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Convolution2D(64, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(64, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(128, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    
    model.add(Convolution2D(256, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(256, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    
    model.add(Flatten())  
    model.add(Dense(1024))  
    model.add(Activation('relu'))  
    model.add(Dropout(0.5))  
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))  
    model.add(Activation('sigmoid'))  
     
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model
    
def cnn_model_experiment_layer_9():
    model = Sequential()  
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, PATCH_SIZE, PATCH_SIZE)))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(32, 3, 3, border_mode='same'))    
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Convolution2D(64, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(64, 3, 3, border_mode='same'))   
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(128, 3, 3, border_mode='same'))   
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    
    model.add(Convolution2D(256, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(256, 3, 3, border_mode='same'))   
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  

    model.add(Convolution2D(512, 3, 3, border_mode='same'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(512, 3, 3, border_mode='same'))   
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    
    model.add(Flatten())  
    model.add(Dense(1024))  
    model.add(Activation('relu'))  
    model.add(Dropout(0.5))  
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    model.add(Dense(1))  
    model.add(Activation('sigmoid'))  
     
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model
    
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image

def combine_features(features):
    num = features.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = features.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=features.dtype)
    for index, img in enumerate(features):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img
    return image
    
    
def main():
    if os.path.exists('test_sample.pkl'):
        with open('test_sample.pkl', 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
    else:
        print 'loading samples'
        x, y = load_dataset()
        print 'loading done'
        print len(y)
    train_x, train_y, val_x, val_y = get_val(x, y)
    model = vgg_like_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(train_x, train_y,                 
                    batch_size=200,         
                    nb_epoch=25,             
                    shuffle=True,                 
                    verbose=2,                     
                    #show_accuracy=True,         
                    validation_data=(val_x, val_y),         
                    callbacks=[early_stopping])
    model.save_weights('practice_weight_vgg.hdf5')
    with open('history.pkl','wb') as f:
        pickle.dump(hist.history,f)

if __name__ == '__main__':
    main()
'''
    if os.path.exists('test_sample.pkl'):
        with open('test_sample.pkl', 'rb') as f:
            dataset = pickle.load(f)
            label = pickle.load(f)
            
    p_index = []
    n_index = []
    for i in xrange(len(label)):
        if label[i] == 0:
            n_index.append(i)
        else:
            p_index.append(i)
    p_sample, p_label = dataset[p_index], label[p_index]
    n_sample, n_label = dataset[n_index], label[n_index]
    
    model = vgg_like_model()
    model.load_weights('practice_weight_vgg.hdf5')
    cnn_predict = Sequential()
    cnn_predict.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 64, 64), 
                    activation = 'relu', weights = model.layers[0].get_weights()))
    cnn_predict.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    #generated_images = cnn_predict.predict(p_sample[:64])
    #image = combine_images(generated_images)
    origin = p_sample[:64]
    Image.fromarray(origin.astype(np.uint8)).save('demo_origin.png')
'''