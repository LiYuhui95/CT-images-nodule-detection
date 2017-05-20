# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:57:45 2017

@author: yuhui
"""
import os
import numpy as np
import dicom
import scipy.ndimage
import cPickle as pickle

NODULE_RECORD_PATH = '/data640/BigNodule/' #LIDC-IDRI-0001_BigNodule/'

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
    
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing, real_resize_factor

def distance (x1, y1, x2, y2):    ##used to calculate the distance between two patch
    temp = abs(x1 - x2) + abs(y1 - y2)
    return temp


def sliding_3D(img, labels, patch_size = 42, patch_thick = 8, stride = 8, stride_thick = 2):
    #nodule_number = np.shape(labels)[0]
    width, height, thickness = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]
    map_width = int((width - patch_size) / stride + 1)   ##sliding iteration times
    map_height = int((height - patch_size) / stride + 1)
    map_thickness = int((thickness - patch_thick) / stride_thick + 1)
    X_negative = np.zeros((map_width, map_height, map_thickness, patch_size, patch_size, patch_thick)).astype('uint8')
    y_negative = np.zeros((map_width, map_height, map_thickness)).astype('uint8')
    for i in xrange(map_width):
        for j in xrange(map_height):
            for k in xrange(map_thickness):
                patch = img[i * stride: i * stride + patch_size,
                            j * stride: j * stride + patch_size,
                            k * stride_thick: k * stride_thick + patch_thick]
                            
                X_negative[i, j, k] = patch            
                x_center = i * stride + patch_size / 2
                y_center = j * stride + patch_size / 2
                z_center = k * stride_thick + patch_thick / 2
                
                dist_xy = distance(labels[:, 0], labels[:, 1], x_center, y_center)
                dist_h = abs(labels[:, 2] - z_center)
                dist = dist_xy + dist_h
                decision = np.where(dist <= 20)[0]  ##threshold of the distance between labels and real point
                if (len(decision) == 0):      ##Label
                    y_negative[i, j] = 0
                else:
                    y_negative[i, j] = 1

    X_negative = X_negative.reshape(-1, patch_size, patch_size, patch_thick)
    y_negative = y_negative.reshape(-1)
    X_negative = X_negative[y_negative == 0]    ##Get negative parts
    y_negative = y_negative[y_negative == 0]
    
    print ("negative examples=", len(y_negative))
    
    X_positive = np.zeros((len(labels) * 3 * 3 * 3, patch_size, patch_size, patch_thick)).astype('uint8')
    y_positive = np.zeros((len(labels) * 3 * 3 * 3)).astype('uint8')
    count = 0
    for i in range(len(labels)):
        x = labels[i, 0]
        y = labels[i, 1]
        z = labels[i, 2]
        #print ("place in the image",x,y)
        for i_offset in range(-4, 5, 4):
            for j_offset in range(-4, 5, 4):
                for k_offset in range(-1, 2, 1):
                    x1 = x + i_offset - patch_size / 2
                    x2 = x + i_offset + patch_size / 2
                    y1 = y + j_offset - patch_size / 2
                    y2 = y + j_offset + patch_size / 2
                    z1 = z + k_offset - patch_thick / 2
                    z2 = z + k_offset + patch_thick / 2
                    if (x1 >= 0 and x2 <= width) and (y1 >= 0 and y2 <= height) and (z1 >= 0 and z2 <= thickness):
                        #patch = img[x1:x2, y1:y2] / 255.0
                        #patch = transform.resize(patch, (64, 64)) * 255
                        #X_positive[count] = patch
                        X_positive[count] = img[x1:x2, y1:y2, z1:z2]
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
    
    
    
    
samples = None
labels = None
loop = 0
for file_name in os.listdir(NODULE_RECORD_PATH):
#file_name = os.listdir(NODULE_RECORD_PATH)[0]
    file_path = NODULE_RECORD_PATH + file_name + '/' + os.listdir(NODULE_RECORD_PATH + file_name)[0]
    print 'locating file path...'
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
            
    # get the core of nodules and slices
    CORE_dic = {}
    core_loc = []
    for z in LOC_dic.keys():
        CORE_dic[z] = []
        for x_min, x_max, y_min, y_max in LOC_dic[z]:
            core = [(x_min+x_max) / 2, (y_min+y_max) / 2]
            CORE_dic[z].append(core)
    
    print CORE_dic
    for i in xrange(len(CORE_dic)): #core_loc = [x_core, y_core, z_min, z_max]
        z = CORE_dic.keys()[i]
        if core_loc == []:
            for x_offset, y_offset in CORE_dic[z]:
                core_loc.append([x_offset, y_offset, z, z])
            continue
        MARK = False
        for x_offset, y_offset in CORE_dic[z]:
            for j in xrange(len(core_loc)):
                discri = distance(core_loc[j][0], core_loc[j][1], x_offset, y_offset)
                if discri <= 50:
                    core_loc[j][0] = (core_loc[j][0] + x_offset) / 2
                    core_loc[j][1] = (core_loc[j][1] + y_offset) / 2
                    core_loc[j][3] = z
                    MARK = True
        if MARK == False:
            core_loc.append([x_offset, y_offset, z, z])
            
    print core_loc
    PATH = '/data640/TCIA-LIDC-IDRI/DOI/'+dicom_path.replace(',','/')
    temp_patient = load_scan(PATH)
    temp_patient_pixels = get_pixels_hu(temp_patient)
    temp_patient_resampled, temp_spacing, resize_factor = resample(temp_patient_pixels, temp_patient, [1,1,1])
    
    for i in xrange(len(temp_patient)):
        for j in xrange(len(core_loc)):
            z_position = float(temp_patient[i].ImagePositionPatient[2])
            if core_loc[j][2] == z_position:
                core_loc[j][2] = i
            if core_loc[j][3] == z_position:
                core_loc[j][3] = i
    
    print core_loc
    for i in xrange(len(core_loc)):
        if core_loc[i][2] <= 0 or core_loc[i][3] <= 0:
            core_loc.pop(i)
    if core_loc == []:
        continue
    
    nodule_labels = np.empty((len(core_loc), 3)).astype('uint8')
    for i in xrange(len(core_loc)):
        nodule_labels[i][0] = core_loc[i][0]
        nodule_labels[i][1] = core_loc[i][1]
        nodule_labels[i][2] = (core_loc[i][2] + core_loc[i][3]) * resize_factor[2] / 2
    
    x, y = sliding_3D(temp_patient_resampled, nodule_labels)
    if samples == None:
        samples = x
        labels = y
    else:
        samples = np.concatenate((samples, x),axis=0)
        labels = np.concatenate((labels,y),axis=0)
    print 'sample number is ', len(labels)
    print len(samples)
    print np.shape(samples)
    #print 'with positive sample number ', labels.count(1)
    loop += 1
    #    if loop == 1:
    #        break
    if loop % 10 == 0:
        with open('samples'+str(loop)+'.pkl','wb') as f:
            pickle.dump(samples, f)
            pickle.dump(labels, f)
        with open(str(loop)+'.txt','w') as f:
            f.write('sample number is ')
            f.write(str(len(labels)))
                #f.write('\n')
                #f.write(str(labels.count(1)))
        samples = None
        labels = None
'''
    for i in xrange(len(temp_patient)):
        z_position = float(temp_patient[i].ImagePositionPatient[2])
        if LOC_dic.has_key(z_position):
            nodule_number = len(LOC_dic[z_position]) / 4
            temp_image = temp_patient_pixels[i]
            m, n = temp_image.shape
            trans_image = np.empty((m,n,3),dtype='uint8')
            trans_image[:,:,0], trans_image[:,:,1], trans_image[:,:,2] = temp_image, temp_image, temp_image
            for j in xrange(nodule_number):
                temp_imgs, temp_labels = selective(trans_image, LOC_dic[z_position][j*4:(j+1)*4])
                positive_index = myfind(1, temp_labels)
                if positive_index:
                    for k in xrange(len(positive_index)):
                        positive_samples.extend([temp_imgs[positive_index[k]]])
                    for k in xrange(len(positive_index)):
                        temp_imgs.pop(positive_index[k])
                        positive_index = [(q-1) for q in positive_index]
                    negative_samples.extend(temp_imgs)
                else:
                    negative_samples.extend(temp_imgs)
    print 'sample number is ', len(positive_samples) + len(negative_samples)
    print 'with positive sample number ', len(positive_samples)
    loop += 1
#    if loop == 1:
#        break
    if loop % 10 == 0:
        positive_number = len(positive_samples)
        random.shuffle(negative_samples)
        with open('samples'+str(loop)+'.pkl','wb') as f:
            pickle.dump(positive_samples, f)
            pickle.dump(negative_samples[:(positive_number*2)], f)
        with open(str(loop)+'.txt','w') as f:
            f.write(str(len(positive_samples) + len(negative_samples)))
            f.write('\n')
            f.write(str(positive_number))
        positive_samples = []
        negative_samples = []
'''