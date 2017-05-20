# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 05:44:12 2017

@author: yuhui
"""
import cPickle as pickle
import os
import numpy as np
import dicom
import selectivesearch
#import matplotlib.pylab as plt
from PIL import Image
import random
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.morphology import binary_dilation, binary_opening
from skimage import measure, transform
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import argparse

THRESHOLD_dist = 16

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")
    
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img
    
# IOU definition
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False #a is the fixed region
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    if if_intersect == True:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1] 
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    #vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    vertice1 = ver1
    area_inter = if_intersection(vertice1[0], vertice1[1], vertice1[2], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = (vertice1[1] - vertice1[0]) * (vertice1[3] - vertice1[2])
        #area_1 = ver1[2] * ver1[3] 
        area_2 = vertice2[4] * vertice2[5] 
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False

def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]
    
def selective(img, ref_rect_int, threshold = 0.3):
    images = []
    labels = []
    img_lbl, regions = selectivesearch.selective_search(
                               img, scale=20, sigma=0.9, min_size=10)
    candidates = set()
    for r in regions: # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        if r['size'] < 120:
            continue
	    # resize for input
        proposal_img, proposal_vertice = clip_pic(img, r['rect'])
	    # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        im = Image.fromarray(proposal_img)
        resized_proposal_img = resize_image(im, 48, 48)
        candidates.add(r['rect'])
        img_float = pil_to_nparray(resized_proposal_img)
        images.append(img_float)
        # IOU
        iou_val = IOU(ref_rect_int, proposal_vertice)
        # labels, let 0 represent default class, which is background
        if iou_val < threshold:
            label = 0
        else:
            label = 1
        labels.append(label)
    return images, labels
    
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

def myfind(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

def ohmy():
    temp_imgs = [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6]
    positive_index = myfind(1, temp_imgs)
    positive_samples = []
    negative_samples = []
    for k in xrange(len(positive_index)):
        positive_samples.extend([temp_imgs[positive_index[k]]])
    for k in xrange(len(positive_index)):
        temp_imgs.pop(positive_index[k])
        positive_index = [(q-1) for q in positive_index]
    negative_samples.extend(temp_imgs)
    return positive_samples, negative_samples

def distance (x1, y1, x2, y2):    ##used to calculate the distance between two patch
    temp = abs(x1 - x2) + abs(y1 - y2)
    return temp
    
def sliding_image_simpler(img, nodule_location, PATCH_SIZE = 64): #PATCH_SIZE should be even in this function
    if np.shape(img)[0] % PATCH_SIZE != 0:
        redundant = np.shape(img)[0] % PATCH_SIZE
        redundant /= 2
        img = img[redundant:(np.shape(img)[0] - redundant),
                  redundant:(np.shape(img)[1] - redundant),]
        for i in xrange(len(nodule_location)):
            nodule_location[i] = [j-redundant for j in nodule_location[i]]
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
    
    X_negative = np.zeros((map_width, map_height, patch_size, patch_size)).astype('uint8') #64,64
    y_negative = np.zeros((map_width, map_height)).astype('uint8')
    for i in xrange(map_width):
        for j in xrange(map_height):
            patch = img[i * stride: i * stride + patch_size,
                        j * stride: j * stride + patch_size]
                        
            X_negative[i, j] = patch            
            x_center = i * stride + patch_size / 2
            y_center = j * stride + patch_size / 2
            
            dist = distance(labels[:, 0], labels[:, 1], x_center, y_center)
            decision = np.where(dist <= THRESHOLD_dist)[0]  ##threshold of the distance between labels and real point
            if (len(decision) == 0):      ##Label
                y_negative[i, j] = 0
            else:
                y_negative[i, j] = 1

    X_negative = X_negative.reshape(-1, patch_size, patch_size) #64,64
    y_negative = y_negative.reshape(-1)
    X_negative = X_negative[y_negative == 0]    ##Get negative parts
    y_negative = y_negative[y_negative == 0]
    
    print ("negative examples=", len(y_negative))
    
    X_positive = np.zeros((len(labels) * 3 * 3, patch_size, patch_size)).astype('uint8') #64,64
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

def main(mode = 'simple', window_size = 64, segment = False):
    loop = 0
    NODULE_RECORD_PATH = '/data640/BigNodule/' #LIDC-IDRI-0001_BigNodule/'
    samples = None
    labels = None
    for file_name in os.listdir(NODULE_RECORD_PATH):
    #file_name = os.listdir(NODULE_RECORD_PATH)[0]
        file_path = NODULE_RECORD_PATH + file_name + '/' + os.listdir(NODULE_RECORD_PATH + file_name)[0]
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
        for i in xrange(len(temp_patient)):
            z_position = float(temp_patient[i].ImagePositionPatient[2])
            if LOC_dic.has_key(z_position):
                temp_image = temp_patient_pixels[i]
                if segment:
                    temp_image = get_segmented_lungs(temp_image)
                if mode == 'simple':
                    temp_sample, temp_label = sliding_image_simpler(temp_image, LOC_dic[z_position], window_size)
                else:
                    temp_sample, temp_label = sliding_image(temp_image, LOC_dic[z_position], window_size, window_size / 4)
                #samples.extend(temp_sample)
                #labels.extend(temp_label)
                if samples == None:
                    samples = temp_sample
                    labels = temp_label
                else:
                    samples = np.concatenate((samples, temp_sample),axis=0)
                    labels = np.concatenate((labels,temp_label),axis=0)
                
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
                f.write('window size is ')
                f.write(str(window_size))
                f.write('\n')
                f.write('sample number is ')
                f.write(str(len(labels)))
                #f.write('\n')
                #f.write(str(labels.count(1)))
            samples = None
            labels = None
            
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--segment", dest="segment", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "simple":
        main(mode = 'simple', window_size = args.window_size, segment = args.segment)
    elif args.mode == "sliding":
        main(mode = 'sliding', window_size = args.window_size, segment = args.segment)
    else:
        main()
    
