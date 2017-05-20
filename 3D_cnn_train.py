# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:10:45 2017

@author: yuhui
"""

import os
import numpy as np
import cPickle as pickle
from keras.models import Sequential
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping

PATCH_SIZE = 42
PATCH_THICK = 8

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
    
    dataset = dataset.reshape(-1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_THICK)
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
    model.add(Convolution3D(32, (3, 3, 3), border_mode='valid', input_shape=(1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_THICK)))  
    model.add(Activation('relu'))  
    model.add(Convolution3D(32, (3, 3, 3)))  
    model.add(Activation('relu'))  
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Convolution3D(64, (3, 3, 3), border_mode='valid'))  
    model.add(Activation('relu'))  
    #model.add(Convolution2D(64, 3, 3))  
    #model.add(Activation('relu'))  
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  
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
    
if __name__ == '__main__':
    if os.path.exists('test_sample.pkl'):
        with open('test_sample.pkl', 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
    else:
        print 'loading samples'
        x, y = load_dataset()
        with open('test_sample.pkl','wb') as f:
            pickle.dump(x,f)
            pickle.dump(y,f)
        print 'loading done'
        print len(y)
    train_x, train_y, val_x, val_y = get_val(x, y)
    print np.shape(train_x)
    print np.shape(train_y)
    model = cnn_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(train_x, train_y,                 
                    #batch_size=200,         
                    nb_epoch=50,             
                    shuffle=True,                 
                    verbose=2,                     
                    #show_accuracy=True,         
                    validation_data=(val_x, val_y),         
                    callbacks=[early_stopping])
    model.save_weights('practice_weight_vgg.hdf5')
    with open('history.pkl','wb') as f:
        pickle.dump(hist.history,f)