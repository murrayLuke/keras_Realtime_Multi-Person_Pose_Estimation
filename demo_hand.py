import cv2
import matplotlib
import pylab as plt
import numpy as np
import util
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from config_reader import config_reader
import scipy
import math
import os
from webcamvideo import WebcamVideoStream
from scipy.ndimage.filters import gaussian_filter

weights_path = "training/NYU_HAND/weights.0005.h5" # weights tarined from scratch 
assert(os.path.exists(weights_path))

def relu(x): 
    return Activation('relu')(x)

def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x):
     
    # Block 1
    x = conv(x, 64, 3, "conv1_1")
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1")
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    
    # Block 3
    x = conv(x, 256, 3, "conv3_1")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_2")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_3")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_4")
    x = relu(x)    
    x = pooling(x, 2, 2, "pool3_1")
    
    # Block 4
    x = conv(x, 512, 3, "conv4_1")
    x = relu(x)    
    x = conv(x, 512, 3, "conv4_2")
    x = relu(x)    
    
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM")
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = relu(x)
    
    return x

def stage1_block(x, num_p, branch):
    
    # Block 1        
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
    
    return x

def stageT_block(x, num_p, stage, branch):
        
    # Block 1        
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
    
    return x

def make_model(num_parts):
    input_shape = (None,None,3)

    img_input = Input(shape=input_shape)

    stages = 6
    np_branch1 = num_parts * 2
    np_branch2 = num_parts

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized)

    # stage 1
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
    return model

def calculateAvgHeatAndPAF(oriImg, param, model_params, num_parts, model):
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_parts))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_parts * 2))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
        print("Input shape: " + str(input_img.shape))  

        output_blobs = model.predict(input_img)
        print("Output shape (heatmap): " + str(output_blobs[1].shape))
        print("Output shape (paf): " + str(output_blobs[0].shape))

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    return heatmap_avg, paf_avg


def showHeatMap(oriImg, param, model_params, num_parts, model):
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_parts))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_parts * 2))
    # first figure shows padded images
    f, axarr = plt.subplots(1, len(multiplier))
    f.set_size_inches((20, 5))
    # second figure shows heatmaps
    f2, axarr2 = plt.subplots(1, len(multiplier))
    f2.set_size_inches((20, 5))
    # third figure shows PAFs
    f3, axarr3 = plt.subplots(2, len(multiplier))
    f3.set_size_inches((20, 10))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
        axarr[m].imshow(imageToTest_padded[:,:,[2,1,0]])
        axarr[m].set_title('Input image: scale %d' % m)

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
        print("Input shape: " + str(input_img.shape))  

        output_blobs = model.predict(input_img)
        print("Output shape (heatmap): " + str(output_blobs[1].shape))
        
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # visualization
        axarr2[m].imshow(oriImg[:,:,[2,1,0]])
        ax2 = axarr2[m].imshow(heatmap[:,:,3], alpha=.5) # right elbow
        axarr2[m].set_title('Heatmaps (Relb): scale %d' % m)
        
        axarr3.flat[m].imshow(oriImg[:,:,[2,1,0]])
        ax3x = axarr3.flat[m].imshow(paf[:,:,16], alpha=.5) # right elbow
        axarr3.flat[m].set_title('PAFs (x comp. of Rwri to Relb): scale %d' % m)
        axarr3.flat[len(multiplier) + m].imshow(oriImg[:,:,[2,1,0]])
        ax3y = axarr3.flat[len(multiplier) + m].imshow(paf[:,:,17], alpha=.5) # right wrist
        axarr3.flat[len(multiplier) + m].set_title('PAFs (y comp. of Relb to Rwri): scale %d' % m)
        
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

        f2.subplots_adjust(right=0.93)
        cbar_ax = f2.add_axes([0.95, 0.15, 0.01, 0.7])
        _ = f2.colorbar(ax2, cax=cbar_ax)

        f3.subplots_adjust(right=0.93)
        cbar_axx = f3.add_axes([0.95, 0.57, 0.01, 0.3])
        _ = f3.colorbar(ax3x, cax=cbar_axx)
        cbar_axy = f3.add_axes([0.95, 0.15, 0.01, 0.3])
        _ = f3.colorbar(ax3y, cax=cbar_axy)
        plt.show()

def visualizeAllParts(frame, heatmap_avg, paf_avg, parts, param):
    all_peaks = []
    peak_counter = 0
    for part, part_name in enumerate(parts):
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
        # plt.imshow(map)
        # plt.imshow(frame[:, :, [2, 1, 0]], alpha=.7)
        # plt.title(f"{part_name}")
        # plt.show()
        
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        plt.plot(peaks_binary)
        plt.show()
        print(peaks_binary.shape)
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    print(peaks_with_score_and_id)

def demo():
    param, model_params = config_reader()
    print(model_params)
    vs = WebcamVideoStream().start()
    num_parts = len(model_params["part_str"])
    model = make_model(num_parts)
    while True :
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('webcam', frame)
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1)
        if key == 27: # esc to quit
            break 
        if key == 32: # space to use image
            print("hit spcae")
        if key == ord('h'):
            showHeatMap(frame, param, model_params, num_parts, model)

        if key == ord('j'):
            avgHeat, avgPaf = calculateAvgHeatAndPAF(frame, param, model_params, num_parts, model)
            visualizeAllParts(frame, avgHeat, avgPaf, model_params["part_str"], param)


    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo()