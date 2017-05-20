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

PATCH_SIZE = 64

def load_dataset(file_path = None):
    print 'start reading files....'
    dataset = None
    label = None
    if file_path == None:
        file_path = os.getcwd()
    for file_name in os.listdir(file_path):
        if file_name.endswith('0.pkl'):
            with open(file_name,'rb') as f:
                tmp_dataset = pickle.load(f)
                tmp_label = pickle.load(f)
            if dataset == None:
                dataset = tmp_dataset
                label = tmp_label
            else:
                dataset = np.concatenate((dataset,tmp_dataset),axis=0)
                label = np.concatenate((label,tmp_label),axis=0)
    
    print 'reading files done'
    print len(label)
    
    dataset = dataset.reshape(-1, 1, PATCH_SIZE, PATCH_SIZE)
    index = [i for i in range(len(dataset))] 
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    
    print 'shuffle done'
    print len(label)
    
    return dataset, label

def get_val(train_x, train_y):
    valid_len = int(len(train_x) / 10)
    #train_x = train_x / np.float32(255.0)
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
    
    
if __name__ == '__main__':
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
    model = cnn_model_experiment()
    if os.path.exists('practice_weight_experiment_7.hdf5'):
        model.load_weights('practice_weight_experiment_7.hdf5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(train_x, train_y,                 
                    batch_size=200,         
                    nb_epoch=75,             
                    shuffle=True,                 
                    verbose=2,                     
                    validation_data=(val_x, val_y),      
                    callbacks=[early_stopping])
    model.save_weights('practice_weight_experiment_7.hdf5')
    with open('history_experiment_7.pkl','wb') as f:
        pickle.dump(hist.history,f)