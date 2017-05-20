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
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import dicom
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.morphology import binary_dilation, binary_opening
from skimage import measure, transform
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi

PATCH_SIZE = 64

def load_scan(path):
    slices = []
    for s in os.listdir(path):
        if s.endswith('dcm'):
            slices.append(dicom.read_file(path + '/' + s))
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in xrange(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
    
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

def sliding_image_simpler(img, nodule_location, PATCH_SIZE = 64):
    PATCH_NUMBER = np.shape(img)[0] / PATCH_SIZE
    x_center = []
    y_center = []
    mark = False
    dataset = []
    label = []
    for x_min, x_max, y_min, y_max in nodule_location:
        x_center.append((x_min + x_max) / 2)
        y_center.append((y_min + y_max) / 2)
    for x in xrange(PATCH_NUMBER):
        for y in xrange(PATCH_NUMBER):
            img_proposal = img[x*PATCH_SIZE:(x+1)*PATCH_SIZE, 
                               y*PATCH_SIZE:(y+1)*PATCH_SIZE]
            dataset.append(img_proposal)
            for i in xrange(len(x_center)):
                if int(x_center[i] / PATCH_SIZE) == x:
                    if int(y_center[i] / PATCH_SIZE) == y:
                        mark = True
            if mark == False:
                label.append(0)
            else:
                label.append(1)
                mark = False
    
    return dataset, label

def sliding_image(img, nodule_location, patch_size = 64, stride = 16):
    nodule_number = len(nodule_location)
    labels = np.empty((nodule_number, 2)).astype('uint8')
    for i in xrange(nodule_number):
        labels[i][0] = (nodule_location[i][0] + nodule_location[i][1]) / 2
        labels[i][1] = (nodule_location[i][2] + nodule_location[i][3]) / 2
        
    width, height = np.shape(img)[0], np.shape(img)[1]
    map_width = int((width - patch_size) / stride + 1)   ##sliding iteration times
    map_height = int((height - patch_size) / stride + 1)
    
    X_negative = np.zeros((map_width, map_height, PATCH_SIZE, PATCH_SIZE)).astype('uint8')
    y_negative = np.zeros((map_width, map_height)).astype('uint8')
    for i in xrange(map_width):
        for j in xrange(map_height):
            patch = img[i * stride: i * stride + patch_size,
                        j * stride: j * stride + patch_size]
                        
            X_negative[i, j] = patch            
            x_center = i * stride + patch_size / 2
            y_center = j * stride + patch_size / 2
            
            dist = distance(labels[:, 0], labels[:, 1], x_center, y_center)
            decision = np.where(dist <= 16)[0]  ##threshold of the distance between labels and real point
            if (len(decision) == 0):      ##Label
                y_negative[i, j] = 0
            else:
                y_negative[i, j] = 1

    X_negative = X_negative.reshape(-1, PATCH_SIZE, PATCH_SIZE)
    y_negative = y_negative.reshape(-1)
    X_negative = X_negative[y_negative == 0]    ##Get negative parts
    y_negative = y_negative[y_negative == 0]
    
    print ("negative examples=", len(y_negative))
    
    X_positive = np.zeros((len(labels) * 3 * 3, PATCH_SIZE, PATCH_SIZE)).astype('uint8')
    y_positive = np.zeros((len(labels) * 3 * 3)).astype('uint8')
    count = 0
    for i in range(len(labels)):
        x = labels[i, 0]
        y = labels[i, 1]
        #print ("place in the image",x,y)
        for i_offset in range(-4, 5, 4):
            for j_offset in range(-4, 5, 4):
                x1 = x + i_offset - patch_size / 2
                x2 = x + i_offset + patch_size / 2
                y1 = y + j_offset - patch_size / 2
                y2 = y + j_offset + patch_size / 2
                if (x1 >= 0 and x2 <= width) and (y1 >= 0 and y2 <= height):
                    #patch = img[x1:x2, y1:y2] / 255.0
                    #patch = transform.resize(patch, (64, 64)) * 255
                    #X_positive[count] = patch
                    X_positive[count] = img[x1:x2, y1:y2]
                    y_positive[count] = 1
                count += 1

    X_positive = X_positive[y_positive == 1]  ##Get positive parts
    y_positive = y_positive[y_positive == 1]
    
    print ("positive examples=", len(y_positive))
    
    indices = np.arange(len(X_negative))   ##Do the random shuffle now
    np.random.shuffle(indices)
    X_negative = X_negative[indices]
    y_negative = y_negative[indices]

    X = np.concatenate([X_negative[:len(y_positive)], X_positive], axis=0) \
        .astype('uint8')
    y = np.concatenate([y_negative[:len(y_positive)], y_positive], axis=0) \
        .astype('uint8')
        
    print("final negative=", sum(y == 0))  ##final check
    print("final positive=", sum(y == 1))
    return X, y
    
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

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def cluster(centers):
    n_class = int(len(centers) * 0.18)
    est = KMeans(n_clusters=n_class, max_iter=1000)
    est.fit(centers)
    new_list = []
    for x, y in est.cluster_centers_:
        min_num = 10000
        min_x = -1
        min_y = -1
        for x_, y_ in centers:
            dist = distance(x, y, x_, y_)
            if (dist < min_num) or (min_x == -1):
                min_num = dist
                min_x = x_
                min_y = y_
        new_list.append([min_x, min_y])
    return new_list
    
def non_max_suppresion(centers, THRESHOLD, WINDOW_SIZE = 64):
    if len(centers) == 0:
        return []
    centers = np.array(centers)
    pick = []
    x1 = centers[:, 0] - WINDOW_SIZE / 2
    y1 = centers[:, 1] - WINDOW_SIZE / 2
    x2 = centers[:, 0] + WINDOW_SIZE / 2
    y2 = centers[:, 1] + WINDOW_SIZE / 2
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = w * h / float(WINDOW_SIZE * WINDOW_SIZE)

        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > THRESHOLD)[0])))

    return centers[pick].astype("int")

def prediction(model, img_batch):
    img_batch = np.reshape(img_batch, (-1, 1, PATCH_SIZE, PATCH_SIZE))
    predict = model.predict(img_batch, verbose=0)
    return np.array(predict)
    
def detection(img, model, model_2 = None, model_3 = None, PATCH_SIZE = 64, stride = 16):
    width, height = np.shape(img)[0], np.shape(img)[1]
    map_width = int((width - PATCH_SIZE) / stride + 1)
    map_height = int((height - PATCH_SIZE) / stride + 1)

    img_batch = np.zeros((map_width, map_height, PATCH_SIZE, PATCH_SIZE)).astype('uint8')
    for i in xrange(map_width):
        for j in xrange(map_width):
            patch = img[i * stride: i * stride + PATCH_SIZE,
                        j * stride: j * stride + PATCH_SIZE]
            img_batch[i, j] = patch

    img_batch = img_batch.reshape(-1,1,PATCH_SIZE,PATCH_SIZE)
    predict_map = prediction(model, img_batch)
    predict_map_2 = prediction(model_2, img_batch)
    predict_map_3 = prediction(model_3, img_batch)
    #print predict_map
    predict_map = predict_map.reshape((map_width, map_height))
    predict_map_2 = predict_map_2.reshape((map_width, map_height))
    predict_map_3 = predict_map_3.reshape((map_width, map_height))
    #print predict_map
    predict_map_final = 0.2 * predict_map + 0.3 * predict_map_2 + 0.5 * predict_map_3

    candicates = []
    img_batch = np.zeros((0, PATCH_SIZE, PATCH_SIZE))
    for i in range(0, map_width):
        for j in range(0, map_width):
            if predict_map_final[i, j] > 0.8:
                patch = img[i * stride: i * stride + PATCH_SIZE,
                            j * stride: j * stride + PATCH_SIZE]
                img_batch = np.append(img_batch, patch.reshape(1, PATCH_SIZE, PATCH_SIZE))
                candicates.append((i * stride + PATCH_SIZE / 2,
                                   j * stride + PATCH_SIZE / 2))
    
    return candicates
    '''
    if model_2 == None:
        predict = prediction(model, img_batch).reshape(-1)
        candicates = np.array(candicates)
        result = candicates #[predict > 0.9]
        return result
    predict = prediction(model_2, img_batch).reshape(-1)
    candicates = np.array(candicates)
    result = candicates[predict > 0.95]
    
    if model_3 == None:
        return result
    
    predict = prediction(model_3, img_batch).reshape(-1)
    candicates = np.array(candicates)
    result = candicates[predict > 0.9]
    return result
    '''

def get_segmented_lungs(im, plot=False, THRESHOLD = -320):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(3, 3, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < THRESHOLD
    if plot == True:
        plots[0,0].axis('off')
        plots[0,0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1,0].axis('off')
        plots[1,0].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2,0].axis('off')
        plots[2,0].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    #print areas
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[0,1].axis('off')
        plots[0,1].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[1,1].axis('off')
        plots[1,1].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(15)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[2,1].axis('off')
        plots[2,1].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[0,2].axis('off')
        plots[0,2].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[1,2].axis('off')
        plots[1,2].imshow(im, cmap=plt.cm.bone) 
        
    return im
    
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
    NODULE_RECORD_PATH = '/data640/BigNodule/' #LIDC-IDRI-0001_BigNodule/'

loop = 0
for file_name in os.listdir(NODULE_RECORD_PATH):
    file_name = os.listdir(NODULE_RECORD_PATH)[0]
    file_path = NODULE_RECORD_PATH +  file_name + '/' +os.listdir(NODULE_RECORD_PATH + file_name)[0]
    print file_path
    number_of_nodule = 0
    with open (file_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            if line[0][0] == 'B':
                number_of_nodule += 1
                
    print 'number of slides which contain big nodules is', number_of_nodule
    LOC_dic = {}
    with open(file_path, 'r') as f:
        dicom_path = f.readline().strip()
        for i in xrange(number_of_nodule):
            temp_location = f.readline().strip().split(',')
            Z_location = float(temp_location[2])
            temp_location = temp_location[4:]
            if len(temp_location) % 2 == 1:
                temp_location = temp_location[:-1]
            temp_location = [int(i) for i in temp_location]
            x_min, x_max = min(temp_location[::2]), max(temp_location[::2])
            y_min, y_max = min(temp_location[1::2]), max(temp_location[1::2])
            if not LOC_dic.has_key(Z_location):
                LOC_dic[Z_location] = []
                LOC_dic[Z_location].append([x_min, x_max, y_min, y_max])
            else:
                LOC_dic[Z_location].append([x_min, x_max, y_min, y_max])
                
    #print LOC_dic
    

    PATH = '/data640/TCIA-LIDC-IDRI/DOI/'+dicom_path.replace(',','/')
    temp_patient = load_scan(PATH)
    temp_patient_pixels = get_pixels_hu(temp_patient)
    for i in range(len(temp_patient)):
        z_position = float(temp_patient[i].ImagePositionPatient[2])
        if LOC_dic.has_key(z_position):
            temp_image = temp_patient_pixels[i]
            #temp_image = get_segmented_lungs(temp_image)
            #temp_sample, temp_label = sliding_image(temp_image, LOC_dic[z_position], 64)
            TEMP_INDEX = i
#            print TEMP_INDEX
#            break
    
#    print np.shape(temp_sample)
#    print np.shape(temp_label)
    
    #model_1 = cnn_model()
    #model_1.load_weights('practice_weight_cnn.hdf5')
            model = vgg_like_model()
            model.load_weights('practice_weight_vgg.hdf5')
            #temp_sample = temp_sample.reshape(-1,1,64,64)
            #y = model.predict(temp_sample)
            #print y
            model_2 = cnn_model_experiment_layer_7()
            model_3 = cnn_model_experiment_layer_9()
            model_2.load_weights('practice_weight_experiment_7.hdf5')
            model_3.load_weights('practice_weight_experiment_9.hdf5')
            centers = detection(temp_image, model,model_2,model_3)#, None, 40)
            print 'center'
            print centers
            centers = cluster(centers)
            
            plt.imshow(temp_patient_pixels[TEMP_INDEX], cmap = plt.cm.bone)
            CurrentAxis = plt.gca()
            for x, y in centers:
                CurrentAxis.add_patch(plt.Rectangle((y - PATCH_SIZE/2, x - PATCH_SIZE/2),PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor='red', linewidth=1))
            centers_min = non_max_suppresion(centers, 0.3)
            for x, y in centers_min:
                CurrentAxis.add_patch(plt.Rectangle((y - PATCH_SIZE/2, x - PATCH_SIZE/2),PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor='blue', linewidth=1))
            for x_min, x_max, y_min, y_max in LOC_dic[z_position]:
                CurrentAxis.add_patch(plt.Rectangle((x_min, y_min),(x_max - x_min),(y_max-y_min),fill=False, edgecolor='yellow', linewidth=1))
            
            plt.savefig('predict' + str(loop)+'.jpg')
            plt.close()
            loop += 1



'''
    temp_sample = np.array(temp_sample)
    temp_sample = temp_sample.reshape(temp_sample.shape[0], -1, 64, 64)
    model = vgg_like_model()
    model.load_weights('practice_weight_vgg.hdf5')
    y = model.predict(temp_sample)
    print y
    plt.imshow(temp_patient_pixels[TEMP_INDEX], cmap = plt.cm.bone)
    CurrentAxis = plt.gca()
    for i in range(len(y)):
        if y[i][0] >= 0.8:
            i += 1
            x_min = i / 8
            y_min = i % 8 
            if y_min == 0:
                y_min = 8
                x_min -= 1
            y_min -= 1
            CurrentAxis.add_patch(plt.Rectangle((x_min*64, y_min*64),64,64,fill=False, edgecolor='red', linewidth=1))
    for x_min, x_max, y_min, y_max in LOC_dic[z_position]:
        CurrentAxis.add_patch(plt.Rectangle((x_min, y_min),(x_max - x_min),(y_max-y_min),fill=False, edgecolor='yellow', linewidth=1))
    
    
    plt.imsave('predict.jpg')
'''

    #final = combine_images(p_sample[:16])
    #plt.imsave(fname = 'demo_origin.jpg', arr = final, cmap = plt.cm.bone)
    #pred = cnn_predict.predict(p_sample[:64])
    #print np.shape(pred)
    #origin = p_sample[:64]
    #final = combine_images(origin)
    #Image.fromarray(final.astype(np.uint8)).save('demo_origin.png')
    #generated_images = cnn_predict.predict(p_sample[:64])
    #image = combine_images(generated_images)
    #Image.fromarray(image.astype(np.uint8)).save('demo2.png')