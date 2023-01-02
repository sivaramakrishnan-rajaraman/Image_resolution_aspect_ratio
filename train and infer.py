
#import libraries

import warnings
warnings.filterwarnings("ignore")

import tensorflow
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import h5py
import time
import skimage.transform
import cv2
import glob
import imageio
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
import scipy as sp
from scipy import ndimage
from skimage import measure, color, io, img_as_ubyte
from skimage.segmentation import clear_border
import random
import csv
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from sklearn.metrics import *
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import skimage.io as io
import skimage.transform as trans
from mpl_toolkits import axes_grid1
from skimage import img_as_uint,img_as_ubyte
import tensorflow as tf
from tensorflow import keras
import math
from math import pi
from math import cos
from math import floor
from tensorflow.keras import backend,layers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import metrics as metrics
from tensorflow.keras.metrics import *
from tensorflow.keras.applications import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.image import *
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
sm.set_framework('tf.keras')
tensorflow.keras.backend.set_image_data_format('channels_last')

#%%
#helper functions
# for interactive plots

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_sample(X, y, preds, binary_preds, ix=None):
    
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10), dpi=500)
    ax[0].set_facecolor('black')
    ax[0].grid(False)
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].set_title('Input')

    ax[1].set_facecolor('black')
    ax[1].grid(False)
    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    ax[1].set_title('GT Mask')

    ax[2].set_facecolor('black')
    ax[2].grid(False)
    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
    ax[2].set_title('Predicted Mask')
    
    ax[3].set_facecolor('black')
    ax[3].grid(False)
    ax[3].imshow(X[ix, ..., 0], cmap='gray')
    ax[3].contour(preds[ix].squeeze(), 
                  linewidths = 4,
                  colors='blue', levels=[0.5])
    ax[3].contour(y[ix].squeeze(), 
                  linewidths = 4,
                  colors='red', levels=[0.5])
    ax[3].set_title('Predicted/ground-truth overlap')
    
# colored masks

MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]

def reshape_arr(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
        
def get_cmap(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        string: [description]
    """
    if arr.ndim == 3:
        return "gray"
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return "jet"
        elif arr.shape[3] == 1:
            return "gray"
        
def mask_to_rgba(mask, color="green"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    assert(color in MASK_COLORS)
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)

def zero_pad_mask(mask, desired_size):
    """[summary]
    
    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask

def plot_imgs(
        org_imgs,
        mask_imgs,
        pred_imgs=None,
        nm_img_to_plot=10,
        figsize=10,
        alpha=0.5,
        color="green"): 
    
    assert(color in MASK_COLORS)

    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False, dpi = 400
    )
    axes[0, 0].set_title("original", fontsize=30)
    axes[0, 1].set_title("ground truth", fontsize=30)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=30)
        axes[0, 3].set_title("overlay", fontsize=30)
    else:
        axes[0, 2].set_title("overlay", fontsize=30)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(
                mask_to_rgba(
                    zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(pred_imgs),
                alpha=alpha,
            )
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(
                mask_to_rgba(
                    zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(mask_imgs),
                alpha=alpha,
            )
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.savefig('overlay.png', format='png', dpi=400)
    plt.show()

#%%
# model evaluation metrics

def dice_coefficient(y_true, y_pred):
    # flatten the image arrays for true and pred
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred[:,:,:,0])

    epsilon=1.0 # to prevent dividing by zero
    return (2*K.sum(y_true*y_pred)+epsilon)/(K.sum(y_true)+K.sum(y_pred)+epsilon)

def dice_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def recall(y_true, y_pred):
    # flatten the image arrays for true and pred
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    return (K.sum(y_true * y_pred)/ (K.sum(y_true) + K.epsilon()))  

def precision(y_true, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    return (K.sum(y_true * y_pred) / (K.sum(y_pred) + K.epsilon()))  

def iou(y_true, y_pred):  #this can be used as a loss if you make it negative
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    union = y_true + ((1 - y_true) * y_pred)
    return (K.sum(y_true * y_pred) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def unique(list1):
    x = np.array(list1)
    print(np.unique(x))  
    
#%%
# image generator functions

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],
                                        new_mask.shape[1]*new_mask.shape[2],
                                        new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255.0
        mask = mask /255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0 # binarize masks
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,
                   mask_folder,aug_dict,
                   image_color_mode = "rgb",
                   mask_color_mode = "grayscale",
                   image_save_prefix  = "image",
                   mask_save_prefix  = "mask",
                   flag_multi_class = False,num_class = 1,
                   save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
     
        
def valGenerator(batch_size,val_path,image_folder,
                 mask_folder,
                 image_color_mode = "rgb",
                 mask_color_mode = "grayscale",
                 image_save_prefix  = "image",
                 mask_save_prefix  = "mask",
                 flag_multi_class = False,num_class = 1,
                 save_to_dir = None,
                 target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

#%%
# print length of the train, validation, and test data
# the height and width should be divisble by 32 
im_height,im_width = 256,256
ids_train = next(os.walk("train_256/image"))[2] # list of names all images in the given path
print("No. of images = ", len(ids_train))
ids_val = next(os.walk("val_256/image"))[2] # list of names all images in the given path
print("No. of images = ", len(ids_val))
ids_test = next(os.walk("test_256/image"))[2] # list of names all images in the given path
print("No. of images = ", len(ids_test))

#%%
threshold=130 #vary
X_ts = np.zeros((len(ids_test), im_height, im_width, 3), dtype=np.float32)
Y_ts = np.zeros((len(ids_test), im_height, im_width, 1), dtype=np.float32)
print(X_ts.shape)
print(Y_ts.shape)
for n, id_ in tqdm(enumerate(ids_test), total=len(ids_test)):
    # Load images
    img = load_img("test_256/image/"+id_, 
                    color_mode = "rgb")
    x_img = img_to_array(img)
    x_img = resize(x_img, (im_height, im_width,3), 
                    mode = 'constant', preserve_range = True)    
    # Load masks
    mask1 = img_to_array(load_img("test_256/label/"+id_, 
                                  color_mode = "grayscale"))
    mask1 = resize(mask1,(im_height, im_width))
    binarized = 1.0 * (mask1 > threshold) #binarize marks
    mask = resize(binarized, (im_height, im_width,1), 
                  mode = 'constant', preserve_range = True)
    X_ts[n] = x_img/255.0
    Y_ts[n] = mask

#%%
print("the unique values from predicted masks is")
unique(Y_ts) 

#%%
#display sample image and mask
ix = random.randint(0, len(X_ts))
has_mask = Y_ts[ix].max() > 0 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))
ax1.imshow(X_ts[ix, ..., 0], cmap = 'gray', interpolation = 'bilinear')
ax1.contour(Y_ts[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
ax1.set_title('CXR')
ax2.imshow(Y_ts[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
ax2.set_title('TB_Mask')

#%%
# declare hyperparameters
n_classes=1 
activation='sigmoid' 
batch_size = 16 
epochs = 128
image_size = im_height
input_size = (im_height,im_width,3)
IN = Input(input_size)

#%%
# loss function
'''
We use the boundary uncertainty evaluation and combine them
with the Focal Tversky loss function
# from https://github.com/mlyg/boundary-uncertainty

'''
def identify_axis(shape):
     # Three dimensional
     if len(shape) == 5 : return [1,2,3]
     # Two dimensional
     elif len(shape) == 4 : return [1,2]
     # Exception - Unknown
     else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
     
def border_uncertainty_sigmoid(seg, alpha = 0.9, beta = 0.1): 
     """
     Parameters
     ----------
     alpha : float, optional
         controls certainty of ground truth inner borders, by default 0.9.
         Higher values more appropriate when over-segmentation is a concern
     beta : float, optional
         controls certainty of ground truth outer borders, by default 0.1
         Higher values more appropriate when under-segmentation is a concern
     """

     res = np.zeros_like(seg)
     check_seg = seg.astype(np.bool)
     
     seg = np.squeeze(seg)

     if check_seg.any():
         kernel = np.ones((3,3),np.uint8)
         im_erode = cv2.erode(seg,kernel,iterations = 1)
         im_dilate = cv2.dilate(seg,kernel,iterations = 1)
         
         # compute inner border and adjust certainty with alpha parameter
         inner = seg - im_erode
         inner = alpha * inner
         # compute outer border and adjust certainty with beta parameter
         outer = im_dilate - seg
         outer = beta * outer
         # combine adjusted borders together with unadjusted image
     
         res = inner + outer + im_erode
         
         res = np.expand_dims(res,axis=-1)

         return res
     else:
         return res

# Enables batch processing of boundary uncertainty 
def border_uncertainty_sigmoid_batch(y_true):
     y_true_numpy = y_true.numpy()
     return np.array([border_uncertainty_sigmoid(y) for y in y_true_numpy]).astype(np.float32)

#focal tversky loss with boundary uncertainty

def focal_tversky_loss_sigmoid(y_true, y_pred, delta=0.7, gamma=0.75, 
                                boundary=True, smooth=0.000001):
    axis = identify_axis(y_true.get_shape())
    if boundary:
        y_true = tf.py_function(func=border_uncertainty_sigmoid_batch, 
                                inp=[y_true], Tout=tf.float32)
    
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Average class scores
    focal_tversky_loss = K.mean(K.pow((1-tversky_class), gamma))
     
    return focal_tversky_loss

#%%
# Declare model
model1 = sm.Unet('inceptionv3',
                  encoder_weights='imagenet',
                  classes=n_classes, 
                    activation=activation) 
model1.summary() 
loss_func = focal_tversky_loss_sigmoid
opt = keras.optimizers.Adam(learning_rate=0.001)
model1.compile(optimizer=opt, 
              loss=loss_func, 
              metrics=['binary_accuracy', 
                        dice_coefficient, 
                        precision, 
                        recall, 
                        iou])                      
callbacks = [EarlyStopping(monitor='val_loss', 
                            patience=10, 
                            verbose=1, 
                            min_delta=1e-4,
                            mode='min'),
              ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1, 
                                patience=5, 
                                verbose=1,
                                min_delta=1e-4, 
                                mode='min'),
              ModelCheckpoint(monitor='val_loss', 
                              filepath='weights/model1.hdf5', 
                              save_best_only=True, 
                              save_weights_only=True, 
                              mode='min', 
                              verbose = 1,
                              save_freq='epoch')]
    
data_gen_args = dict() #vary per requirement                    
myGene = trainGenerator(batch_size,'train_256/',
                        'image','label',                        
                        data_gen_args,
                        target_size = (im_width,im_height), 
                        save_to_dir = None) 
valGene = valGenerator(batch_size,'val_256/',
                        'image','label',
                        target_size = (im_width,im_height), 
                        save_to_dir = None) 
#train
history = model1.fit_generator(generator=myGene,
                      steps_per_epoch=len(ids_train) // batch_size + 1, 
                      epochs=epochs, 
                      callbacks=callbacks, 
                      validation_data=valGene, 
                      validation_steps=len(ids_test) // batch_size + 1, 
                      verbose=1)

print(history.history.keys())

#%%
#plot loss and IoU curves

plt.figure(figsize=(8, 8))
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot( np.argmin(history.history["val_loss"]), 
         np.min(history.history["val_loss"]), 
         marker="x", color="r", 
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()

plt.figure(figsize=(8, 8), dpi=300)
plt.plot(history.history["iou"], label="IoU")
plt.plot(history.history["val_iou"], label="Val_IoU")
plt.plot( np.argmax(history.history["val_iou"]), 
         np.max(history.history["val_iou"]), 
         marker="x", color="r", 
         label="Best model")
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.legend()

#%%
#save the training history
with open('train_history/model_key', 'wb') as file_pi:
    pickle.dump(history.history, file_pi) 

#%%
#Inference: repeat for other models
model1.load_weights("weights/model1.hdf5")
model1.summary()
model1.compile(optimizer=opt, 
              loss=loss_func, 
              metrics=['binary_accuracy', 
                        dice_coefficient, 
                        precision, 
                        recall, 
                        iou])   
# Evaluate on validation set 
score_val = model1.evaluate(X_val, Y_val, verbose=1)
print(model1.metrics_names)
print('Metrics:', score_val)
# Evaluate on the test set 
score_test = model1.evaluate(X_ts, Y_ts, 
                             batch_size = 1,
                             verbose=1)
print(model1.metrics_names)
print('Metrics:', score_test)

#%%
#predict on test data

Y_ts_hat = model1.predict(X_ts, 
                           batch_size=1,
                          verbose=1)
print(Y_ts_hat.shape)
Y_ts_hat_t = (Y_ts_hat > 0.5).astype(np.uint8)

#reshape the predictions 
Y_ts_hat_int = Y_ts_hat.reshape(Y_ts_hat.shape[0]*Y_ts_hat.shape[1]*Y_ts_hat.shape[2], 1)
print(Y_ts_hat_int.shape)
Y_ts_int = Y_ts.reshape(Y_ts.shape[0]*Y_ts.shape[1]*Y_ts.shape[2], 1)
print(Y_ts_int.shape)

#%%
print("the unique values from the ground truth flattened mask is")
unique(Y_ts_int)

print("the unique values from predicted masks is")
unique(Y_ts_hat_int)

#%%
# check quality of predictions using sample CXRs from the test set

plot_sample(X_ts, Y_ts, Y_ts_hat, Y_ts_hat_t, ix=22) # vary ix

#%%
#find optimal segmentation threshold that yields the highest IoU
# from https://www.kaggle.com/code/alexanderliao/deeplabv3/notebook

def iou_metric(y_true, y_pred, print_table=False):
    labels = y_true
    y_pred = y_pred
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), 
                                  bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true, y_pred):
    batch_size = y_true.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true[batch], y_pred[batch])
        metric.append(value)
    return np.mean(metric)

#%%
#find the optimal segmentation threshold

thresholds = np.linspace(0, 1, 200) 
ious = np.array([jaccard_score(Y_ts_int, (Y_ts_hat_int > threshold)) for threshold in tqdm(thresholds)])
plt.figure(figsize=(10,10), dpi=400)
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
plt.plot(thresholds, ious, 
         color='red')
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.plot(threshold_best, iou_best, marker="x", 
         color="blue", 
         label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.legend()

#%%
#measure performance at the optimal segmentation threshold

threshold_confusion = threshold_best
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((Y_ts_hat_int.shape[0]))
for i in range(Y_ts_hat_int.shape[0]):
    if Y_ts_hat_int[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
y_pred1 = np.expand_dims(y_pred, axis=1)
confusion_baseline = confusion_matrix(Y_ts_int, y_pred1)
print (confusion_baseline)
accuracy_baseline = 0
if float(np.sum(confusion_baseline))!=0:
    accuracy_baseline = float(confusion_baseline[0,0]+confusion_baseline[1,1])/float(np.sum(confusion_baseline))
print ("Global Accuracy: " +str(accuracy_baseline))
specificity_baseline = 0
if float(confusion_baseline[0,0]+confusion_baseline[0,1])!=0:
    specificity_baseline = float(confusion_baseline[0,0])/float(confusion_baseline[0,0]+confusion_baseline[0,1])
print ("Specificity: " +str(specificity_baseline))
sensitivity_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[1,0])!=0:
    sensitivity_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[1,0])
print ("Sensitivity: " +str(sensitivity_baseline))
precision_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[0,1])!=0:
    precision_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[0,1])
print ("Precision: " +str(precision_baseline))

#Jaccard similarity index
jaccard_index_baseline = jaccard_score(Y_ts_int, y_pred1)
print ("\nJaccard similarity score: " +str(jaccard_index_baseline))

#F1 score
F1_score_baseline = f1_score(Y_ts_int, y_pred1, 
                    labels=None, 
                    average='binary', sample_weight=None)
print ("\nF1 score: " +str(F1_score_baseline))

#%%
#Area under the ROC curve

fpr, tpr, thresholds = roc_curve(Y_ts_int,
                                 y_pred1)
AUC_ROC = roc_auc_score(Y_ts_int,
                                 y_pred1)
print ("\nArea under the ROC curve : " +str(AUC_ROC))
roc_curve =plt.figure(figsize=(10,10), dpi=50)
plt.plot(fpr,tpr,'-',color="b", 
         label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve',{'fontsize':20})
plt.xlabel("FPR (False Positive Rate)",{'fontsize':20})
plt.ylabel("TPR (True Positive Rate)",{'fontsize':20})
plt.legend(loc="lower right")

#%%
#mAP or precision recall curve

precision, recall, thresholds = precision_recall_curve(Y_ts_int,
                                 y_pred1)
precision = np.fliplr([precision])[0] 
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure(figsize=(15,10), dpi=40)
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(recall,precision,'-',color="r",
          label='Area Under the Curve (AUC = %0.4f) ' % AUC_prec_rec)
plt.title('Precision - Recall curve',{'fontsize':20})
plt.xlabel("Recall",{'fontsize':20})
plt.ylabel("Precision",{'fontsize':20})
plt.legend(loc="lower right")
#plt.savefig("./models/pr.png")

#%%
'''
Test time augmentation is a common way to improve the 
accuracy of image classifiers especially in the 
case of deep learning. We change the image we
 want to predict in some ways, get the predictions 
 for all of these images and average the predictions. 
 The intuition behind this is that even if the test 
 image is not too easy to make a prediction, the 
 transformations change it such that the model has 
 higher chances of capturing the target shape and 
 predicting accordingly. 
'''

#%%
# from https://stepup.ai/test_time_data_augmentation/#:~:text=Test%2DTime%20Data%20Augmentation%20(short,use%20with%20deep%20learning%20models.&text=the%20predictions%20are%20then%20aggregated%20to%20get%20a%20higher%20overall%20accuracy.

def flip_lr(images):
    return np.flip(images, axis=2)

def shift(images, shift, axis):
    return np.roll(images, shift, axis=axis)

def rotate(images, angle):
    return sp.ndimage.rotate(
        images, angle, axes=(1,2),
        reshape=False, mode='nearest')

#%%
pred = model1.predict(X_ts, verbose=1)
pred_int = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2], 1)

pred_f0 = model1.predict(flip_lr(X_ts))
pred_f0_int = pred_f0.reshape(pred_f0.shape[0]*pred_f0.shape[1]*pred_f0.shape[2], 1)

pred_w0 = model1.predict(shift(X_ts, -5, axis=2))
pred_w1 = model1.predict(shift(X_ts, 5, axis=2))
pred_w0_int = pred_w0.reshape(pred_w0.shape[0]*pred_w0.shape[1]*pred_w0.shape[2], 1)
pred_w1_int = pred_w1.reshape(pred_w1.shape[0]*pred_w1.shape[1]*pred_w1.shape[2], 1)

pred_h0 = model1.predict(shift(X_ts, -5, axis=1)) #-3 ans +3
pred_h1 = model1.predict(shift(X_ts, 5, axis=1))
pred_h0_int = pred_h0.reshape(pred_h0.shape[0]*pred_h0.shape[1]*pred_h0.shape[2], 1)
pred_h1_int = pred_h1.reshape(pred_h1.shape[0]*pred_h1.shape[1]*pred_h1.shape[2], 1)

pred_r0 = model1.predict(rotate(X_ts, -5))
pred_r1 = model1.predict(rotate(X_ts, 5))
pred_r0_int = pred_r0.reshape(pred_r0.shape[0]*pred_r0.shape[1]*pred_r0.shape[2], 1)
pred_r1_int = pred_r1.reshape(pred_r1.shape[0]*pred_r1.shape[1]*pred_r1.shape[2], 1)

#%%
#combine non-augmented (original) and flip-lr
pred_flip = (pred_int+pred_f0_int)/2

#combine non-augmented and width shift
pred_width = (pred_int+pred_w0_int+pred_w1_int)/3

#combine non-augmented and height shift
pred_height = (pred_int+pred_h0_int+pred_h1_int)/3

#combine non-augmented width, and height shift
pred_width_height = (pred_int+pred_h0_int+pred_h1_int+pred_w0_int+pred_w1_int)/5

#combine non-augmented width, flip and height shift
pred_width_height_flip = (pred_int+pred_h0_int+pred_h1_int+\
                          pred_w0_int+pred_w1_int+\
                              pred_f0_int)/6

#combine non-augmented and rotation
pred_rotate = (pred_int+pred_r0_int+pred_r1_int)/3

#combine everything
pred_all = (pred_int+pred_h0_int+pred_h1_int+\
                              pred_w0_int+pred_w1_int+\
                                  pred_f0_int+\
                                      pred_r0_int+\
                                      pred_r1_int)/8

pred_all1 = (pred_int+pred_h0_int+pred_h1_int+\
                                  pred_w0_int+pred_w1_int+\
                                      pred_r0_int+\
                                          pred_r1_int)/7
#%%
#find optimal threshold as before for each case
#repeat for each case of augmentation combination

thresholds = np.linspace(0, 1, 200)
ious = np.array([jaccard_score(Y_ts_int, (pred_flip > threshold)) for threshold in tqdm(thresholds)])
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

#%%
threshold_confusion = threshold_best
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((pred_flip.shape[0]))
for i in range(pred_flip.shape[0]):
    if pred_flip[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
y_pred1 = np.expand_dims(y_pred, axis=1)
confusion_baseline = confusion_matrix(Y_ts_int, y_pred1)
print (confusion_baseline)
accuracy_baseline = 0
if float(np.sum(confusion_baseline))!=0:
    accuracy_baseline = float(confusion_baseline[0,0]+confusion_baseline[1,1])/float(np.sum(confusion_baseline))
print ("Global Accuracy: " +str(accuracy_baseline))
specificity_baseline = 0
if float(confusion_baseline[0,0]+confusion_baseline[0,1])!=0:
    specificity_baseline = float(confusion_baseline[0,0])/float(confusion_baseline[0,0]+confusion_baseline[0,1])
print ("Specificity: " +str(specificity_baseline))
sensitivity_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[1,0])!=0:
    sensitivity_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[1,0])
print ("Sensitivity: " +str(sensitivity_baseline))
precision_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[0,1])!=0:
    precision_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[0,1])
print ("Precision: " +str(precision_baseline))

#Jaccard similarity index
jaccard_index_baseline = jaccard_score(Y_ts_int, y_pred1)
print ("\nJaccard similarity score: " +str(jaccard_index_baseline))

#F1 score
F1_score_baseline = f1_score(Y_ts_int, y_pred1, 
                    labels=None, 
                    average='binary', sample_weight=None)
print ("\nF1 score: " +str(F1_score_baseline))


#%%
#save model predictions

im_height,im_width = 256,256
source_i = glob.glob("test_256/image/*.png")
source_i.sort()
source_l = glob.glob("test_256/label/*.png")
source_l.sort()
threshold_i = 0.07035 # toptimal segmentation threshold, varies per model
threshold_l = 0.5
model1.load_weights("weights/model1.hdf5")
model1.summary()

for f1,f2 in zip(source_i,source_l):
    img_i = load_img(f1,color_mode = "rgb") 
    img_l = load_img(f2,color_mode = "grayscale")
    img_name = f1.split(os.sep)[-1] 
    
    #preprocess the image and label
    img_i = img_i.resize((im_width,im_height))
    img_l = img_l.resize((im_width,im_height))
    x_i = img_to_array(img_i)
    x_l = img_to_array(img_l)
    x_i = x_i.astype('float32') / 255 
    x1 = np.expand_dims(x_i, axis=0) 
    
    #predict on the image
    pred_i = model1.predict(x1) 
    
    #binarize the prediction based on the optimal segmentation threshold
    binarized_i = 1.0 * (pred_i > threshold_i) 
    
    #binarize label
    binarized_l = 1.0 * (x_l > threshold_l)
      
    # resize the ground truth label and the predicted label  
    #ground truth label
    mask_l = resize(binarized_l, (im_height, im_width,1), 
                  mode = 'constant', preserve_range = True)
    #predicted label
    mask_i = np.reshape(binarized_i,(im_height,im_width,1))
        
    #write to a image file
    #ground truth binarized
    imageio.imwrite('256_c/gt/{}.png'.format(img_name[:-4]), 
                    mask_l)
    #prediction binarized
    imageio.imwrite('256_c/pred/{}.png'.format(img_name[:-4]), 
                    mask_i)

#%%
# create a snapshot ensemble with custom learning rate schedule
 
class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=1): #hyperparamters to vary
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    # calculate learning rate for epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs={}):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)

    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # check if we can save model
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # save model to file
            filename = "weights/snapshot256u_c_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            print('>saved snapshot %s, epoch %d' % (filename, epoch))

#%%
# create snapshot ensemble callback
# Vary Cycle Length. Use a shorter or longer cycle length and compare results.
# Vary Maximum Learning Rate. Use a larger or smaller maximum learning rate and compare results.
batch_size = 16
n_epochs = 320
n_cycles = n_epochs / 40  
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.001) 
callbacks_list = [ca]

#%%
# model architecture and loss function
model1 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation) #FPN, Linknet,for psp set input shape as 258//6 should be zero
opt = keras.optimizers.Adam()
loss_func=focal_tversky_loss_sigmoid,
model1.compile(optimizer=opt, 
              loss=loss_func, 
              metrics=['binary_accuracy', 
                       dice_coefficient, 
                       precision, 
                       recall, 
                        iou])  

#%%
#train the UNET model
t=time.time() 
print('-'*30)
print('Start Training the model...')
print('-'*30)

data_gen_args = dict() #varies for the problem

myGene = trainGenerator(batch_size,'train_256/',
                        'image','label',                        
                        data_gen_args,
                        target_size = (im_height,im_width),
                        save_to_dir = None) 
valGene = valGenerator(batch_size,'val_256/',
                       'image','label',
                       target_size = (im_height,im_width),                       
                       save_to_dir = None) 
#train
history = model1.fit_generator(generator=myGene,
                     steps_per_epoch=len(ids_train) // batch_size + 1, #Add +1 if not divisible
                     epochs=n_epochs, 
                     callbacks=callbacks_list,
                     validation_data=valGene, 
                     validation_steps=len(ids_test) // batch_size + 1, #Add +1 if not divisible
                     verbose=1)

print('Training time: %s' % (time.time()-t))
print(history.history.keys())

#%% 
# do inference with each model snbapshot as before

#%%
'''
Averaging the predictions obtained with the best
test time augmentation method for each of the top-3 snapshots, 
and saving the predictions. 
Repeat for other top-4, top-5, and top-6 snapshots, and then see its performance

'''
im_height,im_width = 256,256
source_i = glob.glob("test_256/image/*.png")
source_i.sort()
source_l = glob.glob("test_256/label/*.png")
source_l.sort()

model1 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation)
model2 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation)
model3 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation)
model4 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation)
model5 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation)
model6 = sm.Unet('inceptionv3', 
                      encoder_weights='imagenet', 
                      classes=n_classes, 
                      activation=activation)

model1.load_weights("weights/snapshot256u_c_2.h5")
model1.summary()
model2.load_weights("weights/snapshot256u_c_3.h5")
model2.summary()
model3.load_weights("weights/snapshot256u_c_5.h5")
model3.summary()
model4.load_weights("weights/snapshot256u_c_7.h5")
model4.summary()
model5.load_weights("weights/snapshot256u_c_6.h5")
model5.summary()
model6.load_weights("weights/snapshot256u_c_4.h5")
model6.summary()

#optimal threshold
threshold_top2 = 0.5779
threshold_top3 = 0.5126
threshold_top4 = 0.4925
threshold_top5 = 0.4874
threshold_top6 = 0.4925
threshold_l = 0.5

for f1,f2 in zip(source_i,source_l):
    img_i = load_img(f1,color_mode = "rgb") 
    img_l = load_img(f2,color_mode = "grayscale")
    img_name = f1.split(os.sep)[-1]  
    
    #preprocess the image and label
    img_i = img_i.resize((im_width,im_height))
    img_l = img_l.resize((im_width,im_height))
    x_i = img_to_array(img_i)
    x_l = img_to_array(img_l)
    
    #binarize label
    binarized_l = 1.0 * (x_l > threshold_l)  
    
    # resize the ground truth label and the predicted label  
    #ground truth label
    mask_l = resize(binarized_l, (im_height, im_width,1), 
                  mode = 'constant', preserve_range = True)
    
    x_i = x_i.astype('float32') / 255 
    x1 = np.expand_dims(x_i, axis=0) 
    
    #predict on the image
    pred1 = model1.predict(x1)
    pred1_int = pred1.reshape(pred1.shape[0]*pred1.shape[1]*pred1.shape[2], 1)
    pred2 = model2.predict(x1)
    pred2_int = pred2.reshape(pred2.shape[0]*pred2.shape[1]*pred2.shape[2], 1)
    pred3 = model3.predict(x1)
    pred3_int = pred3.reshape(pred3.shape[0]*pred3.shape[1]*pred3.shape[2], 1)
    pred4 = model4.predict(x1)
    pred4_int = pred4.reshape(pred4.shape[0]*pred4.shape[1]*pred4.shape[2], 1)
    pred5 = model5.predict(x1)
    pred5_int = pred5.reshape(pred5.shape[0]*pred5.shape[1]*pred5.shape[2], 1)
    pred6 = model6.predict(x1)
    pred6_int = pred6.reshape(pred6.shape[0]*pred6.shape[1]*pred6.shape[2], 1)
    
    #flip left right 
    pred1_f0 = model1.predict(flip_lr(x1))
    pred1_f0_int = pred1_f0.reshape(pred1_f0.shape[0]*pred1_f0.shape[1]*pred1_f0.shape[2], 1)
    
    pred2_f0 = model2.predict(flip_lr(x1))
    pred2_f0_int = pred2_f0.reshape(pred2_f0.shape[0]*pred2_f0.shape[1]*pred2_f0.shape[2], 1)
    
    pred3_f0 = model3.predict(flip_lr(x1))
    pred3_f0_int = pred3_f0.reshape(pred3_f0.shape[0]*pred3_f0.shape[1]*pred3_f0.shape[2], 1)
    
    pred4_f0 = model4.predict(flip_lr(x1))
    pred4_f0_int = pred4_f0.reshape(pred4_f0.shape[0]*pred4_f0.shape[1]*pred4_f0.shape[2], 1)
    
    pred5_f0 = model5.predict(flip_lr(x1))
    pred5_f0_int = pred5_f0.reshape(pred5_f0.shape[0]*pred5_f0.shape[1]*pred5_f0.shape[2], 1)
    
    pred6_f0 = model6.predict(flip_lr(x1))
    pred6_f0_int = pred6_f0.reshape(pred6_f0.shape[0]*pred6_f0.shape[1]*pred6_f0.shape[2], 1)
    
    # width and height shifting
    pred1_w0 = model1.predict(shift(x1, -5, axis=2))
    pred1_w1 = model1.predict(shift(x1, 5, axis=2))
    pred1_w0_int = pred1_w0.reshape(pred1_w0.shape[0]*pred1_w0.shape[1]*pred1_w0.shape[2], 1)
    pred1_w1_int = pred1_w1.reshape(pred1_w1.shape[0]*pred1_w1.shape[1]*pred1_w1.shape[2], 1)
    
    pred2_w0 = model2.predict(shift(x1, -5, axis=2))
    pred2_w1 = model2.predict(shift(x1, 5, axis=2))
    pred2_w0_int = pred2_w0.reshape(pred2_w0.shape[0]*pred2_w0.shape[1]*pred2_w0.shape[2], 1)
    pred2_w1_int = pred2_w1.reshape(pred2_w1.shape[0]*pred2_w1.shape[1]*pred2_w1.shape[2], 1)
    
    pred3_w0 = model3.predict(shift(x1, -5, axis=2))
    pred3_w1 = model3.predict(shift(x1, 5, axis=2))
    pred3_w0_int = pred3_w0.reshape(pred3_w0.shape[0]*pred3_w0.shape[1]*pred3_w0.shape[2], 1)
    pred3_w1_int = pred3_w1.reshape(pred3_w1.shape[0]*pred3_w1.shape[1]*pred3_w1.shape[2], 1)
    
    pred4_w0 = model4.predict(shift(x1, -5, axis=2))
    pred4_w1 = model4.predict(shift(x1, 5, axis=2))
    pred4_w0_int = pred4_w0.reshape(pred4_w0.shape[0]*pred4_w0.shape[1]*pred4_w0.shape[2], 1)
    pred4_w1_int = pred4_w1.reshape(pred4_w1.shape[0]*pred4_w1.shape[1]*pred4_w1.shape[2], 1)
    
    pred5_w0 = model5.predict(shift(x1, -5, axis=2))
    pred5_w1 = model5.predict(shift(x1, 5, axis=2))
    pred5_w0_int = pred5_w0.reshape(pred5_w0.shape[0]*pred5_w0.shape[1]*pred5_w0.shape[2], 1)
    pred5_w1_int = pred5_w1.reshape(pred5_w1.shape[0]*pred5_w1.shape[1]*pred5_w1.shape[2], 1)
    
    pred6_w0 = model6.predict(shift(x1, -5, axis=2))
    pred6_w1 = model6.predict(shift(x1, 5, axis=2))
    pred6_w0_int = pred6_w0.reshape(pred6_w0.shape[0]*pred6_w0.shape[1]*pred6_w0.shape[2], 1)
    pred6_w1_int = pred6_w1.reshape(pred6_w1.shape[0]*pred6_w1.shape[1]*pred6_w1.shape[2], 1)
    
    pred1_h0 = model1.predict(shift(x1, -5, axis=1)) #-3 ans +3
    pred1_h1 = model1.predict(shift(x1, 5, axis=1))
    pred1_h0_int = pred1_h0.reshape(pred1_h0.shape[0]*pred1_h0.shape[1]*pred1_h0.shape[2], 1)
    pred1_h1_int = pred1_h1.reshape(pred1_h1.shape[0]*pred1_h1.shape[1]*pred1_h1.shape[2], 1)
    
    pred2_h0 = model2.predict(shift(x1, -5, axis=1)) #-3 ans +3
    pred2_h1 = model2.predict(shift(x1, 5, axis=1))
    pred2_h0_int = pred2_h0.reshape(pred2_h0.shape[0]*pred2_h0.shape[1]*pred2_h0.shape[2], 1)
    pred2_h1_int = pred2_h1.reshape(pred2_h1.shape[0]*pred2_h1.shape[1]*pred2_h1.shape[2], 1)
    
    pred3_h0 = model3.predict(shift(x1, -5, axis=1)) #-3 ans +3
    pred3_h1 = model3.predict(shift(x1, 5, axis=1))
    pred3_h0_int = pred3_h0.reshape(pred3_h0.shape[0]*pred3_h0.shape[1]*pred3_h0.shape[2], 1)
    pred3_h1_int = pred3_h1.reshape(pred3_h1.shape[0]*pred3_h1.shape[1]*pred3_h1.shape[2], 1)
    
    pred4_h0 = model4.predict(shift(x1, -5, axis=1)) #-3 ans +3
    pred4_h1 = model4.predict(shift(x1, 5, axis=1))
    pred4_h0_int = pred4_h0.reshape(pred4_h0.shape[0]*pred4_h0.shape[1]*pred4_h0.shape[2], 1)
    pred4_h1_int = pred4_h1.reshape(pred4_h1.shape[0]*pred4_h1.shape[1]*pred4_h1.shape[2], 1)
    
    pred5_h0 = model5.predict(shift(x1, -5, axis=1)) #-3 ans +3
    pred5_h1 = model5.predict(shift(x1, 5, axis=1))
    pred5_h0_int = pred5_h0.reshape(pred5_h0.shape[0]*pred5_h0.shape[1]*pred5_h0.shape[2], 1)
    pred5_h1_int = pred5_h1.reshape(pred5_h1.shape[0]*pred5_h1.shape[1]*pred5_h1.shape[2], 1)
    
    pred6_h0 = model6.predict(shift(x1, -5, axis=1)) #-3 ans +3
    pred6_h1 = model6.predict(shift(x1, 5, axis=1))
    pred6_h0_int = pred6_h0.reshape(pred6_h0.shape[0]*pred6_h0.shape[1]*pred6_h0.shape[2], 1)
    pred6_h1_int = pred6_h1.reshape(pred6_h1.shape[0]*pred6_h1.shape[1]*pred6_h1.shape[2], 1)
    
    
    
    #rotation   
    pred1_r0 = model1.predict(rotate(x1, -5))
    pred1_r1 = model1.predict(rotate(x1, 5))
    pred1_r0_int = pred1_r0.reshape(pred1_r0.shape[0]*pred1_r0.shape[1]*pred1_r0.shape[2], 1)
    pred1_r1_int = pred1_r1.reshape(pred1_r1.shape[0]*pred1_r1.shape[1]*pred1_r1.shape[2], 1)
    
    pred2_r0 = model2.predict(rotate(x1, -5))
    pred2_r1 = model2.predict(rotate(x1, 5))
    pred2_r0_int = pred2_r0.reshape(pred2_r0.shape[0]*pred2_r0.shape[1]*pred2_r0.shape[2], 1)
    pred2_r1_int = pred2_r1.reshape(pred2_r1.shape[0]*pred2_r1.shape[1]*pred2_r1.shape[2], 1)
    
    pred3_r0 = model3.predict(rotate(x1, -5))
    pred3_r1 = model3.predict(rotate(x1, 5))
    pred3_r0_int = pred3_r0.reshape(pred3_r0.shape[0]*pred3_r0.shape[1]*pred3_r0.shape[2], 1)
    pred3_r1_int = pred3_r1.reshape(pred3_r1.shape[0]*pred3_r1.shape[1]*pred3_r1.shape[2], 1)
    
    pred4_r0 = model4.predict(rotate(x1, -5))
    pred4_r1 = model4.predict(rotate(x1, 5))
    pred4_r0_int = pred4_r0.reshape(pred4_r0.shape[0]*pred4_r0.shape[1]*pred4_r0.shape[2], 1)
    pred4_r1_int = pred4_r1.reshape(pred4_r1.shape[0]*pred4_r1.shape[1]*pred4_r1.shape[2], 1)
    
    pred5_r0 = model5.predict(rotate(x1, -5))
    pred5_r1 = model5.predict(rotate(x1, 5))
    pred5_r0_int = pred5_r0.reshape(pred5_r0.shape[0]*pred5_r0.shape[1]*pred5_r0.shape[2], 1)
    pred5_r1_int = pred5_r1.reshape(pred5_r1.shape[0]*pred5_r1.shape[1]*pred5_r1.shape[2], 1)
        
    pred6_r0 = model6.predict(rotate(x1, -5))
    pred6_r1 = model6.predict(rotate(x1, 5))
    pred6_r0_int = pred6_r0.reshape(pred6_r0.shape[0]*pred6_r0.shape[1]*pred6_r0.shape[2], 1)
    pred6_r1_int = pred6_r1.reshape(pred6_r1.shape[0]*pred6_r1.shape[1]*pred6_r1.shape[2], 1)
    
    #for the top snapshot S2, pred-height
    pred_height1 = (pred1_int+pred1_h0_int+pred1_h1_int)/3
    
    # for the second top snapshot S3, pred-all
    pred_all2 = (pred2_int+pred2_h0_int+pred2_h1_int+\
                                  pred2_w0_int+pred2_w1_int+\
                                      pred2_f0_int+\
                                          pred2_r0_int+\
                                          pred2_r1_int)/8
        
    # for the third top snapshot S5, pred-all
    pred_all3 = (pred3_int+pred3_h0_int+pred3_h1_int+\
                                  pred3_w0_int+pred3_w1_int+\
                                      pred3_f0_int+\
                                          pred3_r0_int+\
                                          pred3_r1_int)/8
       
        
    # for the fourth top snapshot S7, pred-all
    pred_all4 = (pred4_int+pred4_h0_int+pred4_h1_int+\
                                  pred4_w0_int+pred4_w1_int+\
                                      pred4_f0_int+\
                                          pred4_r0_int+\
                                          pred4_r1_int)/8
        
    # for the fifth top snapshot S6, pred-width-height
    pred_width_height5 = (pred5_int+pred5_h0_int+\
                          pred5_h1_int+pred5_w0_int+pred5_w1_int)/5   
        
    # for the sixth top snapshot S4, pred-all
    pred_all6 = (pred6_int+pred6_h0_int+pred6_h1_int+\
                                  pred6_w0_int+pred6_w1_int+\
                                      pred6_f0_int+\
                                          pred6_r0_int+\
                                          pred6_r1_int)/8    
 
    # Next doing an average of the top-N (N=3, repeat other) snapshots  
    # img_AVG = (pred_height1+pred_all2)/2 
    img_AVG = (pred_height1+pred_all2+pred_all3)/3 
    # img_AVG = (pred_height1+pred_all2+pred_all3+pred_all4)/4
    # img_AVG = (pred_height1+pred_all2+pred_all3+pred_all4+pred_width_height5)/5  
    # img_AVG = (pred_height1+pred_all2+pred_all3+\
    #             pred_all4+pred_width_height5+\
    #             pred_all6)/6 
    for i in range(img_AVG.shape[0]): 
        if img_AVG[i]>=threshold_top6:
            img_AVG[i]=1
        else:
            img_AVG[i]=0    
    mask_i = np.reshape(img_AVG,(im_height,im_width,1))     
        
    #write to a image file
    #ground truth binarized
    imageio.imwrite('snapshot/avg3/gt/{}.png'.format(img_name[:-4]), 
                    mask_l)
    #prediction binarized
    imageio.imwrite('snapshot/avg3/pred/{}.png'.format(img_name[:-4]), 
                    mask_i)   
#%%
#perform inference as before using the optimal segmentation threshold

ids_test = next(os.walk("snapshot/avg3"))[2] 
print("No. of images = ", len(ids_test))

Y_ts_hat = np.zeros((len(ids_test), im_height, im_width, 1), 
                    dtype=np.float32)
print(Y_ts_hat.shape)

for n, id_ in tqdm(enumerate(ids_test), total=len(ids_test)):
    # Load masks
    mask1 = img_to_array(load_img("snapshot/avg3/"+id_, 
                                  color_mode = "grayscale")) #grayscale  
    mask = resize(mask1, (im_height, im_width,1), 
                  mode = 'constant', preserve_range = True)
    # Save images
    Y_ts_hat[n] = mask/255
    
#%%
#reshape the predictions 

Y_ts_hat_int = Y_ts_hat.reshape(Y_ts_hat.shape[0]*Y_ts_hat.shape[1]*Y_ts_hat.shape[2], 1)
print(Y_ts_hat_int.shape)
Y_ts_int = Y_ts.reshape(Y_ts.shape[0]*Y_ts.shape[1]*Y_ts.shape[2], 1)
print(Y_ts_int.shape)

#%%
print("the unique values from the ground truth flattened mask is")
unique(Y_ts_int)

print("the unique values from predicted masks is")
unique(Y_ts_hat_int)

#%%
#find optimal segmentation threshold

thresholds = np.linspace(0, 1, 200)
ious = np.array([jaccard_score(Y_ts_int, (Y_ts_hat_int > threshold)) for threshold in tqdm(thresholds)])
plt.figure(figsize=(10,10), dpi=400)
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
plt.plot(thresholds, ious, 
         color='red')
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.plot(threshold_best, iou_best, marker="x", 
         color="blue", 
         label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.legend()

#%%
#Measure performance at the optimal threshold

threshold_confusion = threshold_best
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((Y_ts_hat_int.shape[0]))
for i in range(Y_ts_hat_int.shape[0]):
    if Y_ts_hat_int[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
y_pred1 = np.expand_dims(y_pred, axis=1)
confusion_baseline = confusion_matrix(Y_ts_int, y_pred1)
print (confusion_baseline)
accuracy_baseline = 0
if float(np.sum(confusion_baseline))!=0:
    accuracy_baseline = float(confusion_baseline[0,0]+confusion_baseline[1,1])/float(np.sum(confusion_baseline))
print ("Global Accuracy: " +str(accuracy_baseline))
specificity_baseline = 0
if float(confusion_baseline[0,0]+confusion_baseline[0,1])!=0:
    specificity_baseline = float(confusion_baseline[0,0])/float(confusion_baseline[0,0]+confusion_baseline[0,1])
print ("Specificity: " +str(specificity_baseline))
sensitivity_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[1,0])!=0:
    sensitivity_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[1,0])
print ("Sensitivity: " +str(sensitivity_baseline))
precision_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[0,1])!=0:
    precision_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[0,1])
print ("Precision: " +str(precision_baseline))

#Jaccard similarity index
jaccard_index_baseline = jaccard_score(Y_ts_int, y_pred1)
print ("\nJaccard similarity score: " +str(jaccard_index_baseline))

#F1 score
F1_score_baseline = f1_score(Y_ts_int, y_pred1, 
                    labels=None, 
                    average='binary', sample_weight=None)
print ("\nF1 score: " +str(F1_score_baseline))

#%%