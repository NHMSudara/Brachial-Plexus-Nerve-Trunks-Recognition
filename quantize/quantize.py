import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
from numpy import asarray
from sklearn.model_selection import train_test_split
from glob import glob
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.optimizers import Adam


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1):
    m = len(img)                     # number of images
    i_h, i_w, i_c = target_shape_img   # pull height, width, and channels of image
    m_h, m_w, m_c = target_shape_mask  # pull height, width, and channels of mask

    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    # Resize images and masks
    for idx, file in enumerate(img):
        # Convert image into an array of desired shape (3 channels)
        path = os.path.join( file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h, i_w))
        single_img = np.array(single_img, dtype=np.float32) / 255.0
        X[idx] = single_img

        # Convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[idx]
        path = os.path.join( single_mask_ind)
        single_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(single_mask)
        single_mask = cv2.resize(single_mask, (m_h, m_w), interpolation=cv2.INTER_NEAREST)
        single_mask = np.expand_dims(single_mask, axis=-1)  # Ensure the mask has a channel dimension
        y[idx] = single_mask

    return X, y

smooth = 1
def dice_coef(y_true, y_pred):
    y_true_float32 = tf.cast(y_true, tf.float32)
    y_pred_float32 = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()( y_true_float32)
    y_pred = tf.keras.layers.Flatten()(y_pred_float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# Define target shapes for images and masks
target_shape_img = [96, 96, 3]
target_shape_mask = [96, 96, 1]

# Load and preprocess the training data
train_path = 'test_1'  # Change this to your actual training data path
train_x, train_y = load_data(train_path)
print(train_x)
X_train, Y_train = PreprocessData(train_x, train_y, target_shape_img, target_shape_mask, train_path)

# Split the data into training and calibration datasets
X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# Convert the calibration data to a TensorFlow dataset
calib_dataset = tf.data.Dataset.from_tensor_slices(X_calib).batch(1)

custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef}


# Load your pre-trained Keras model
model = tf.keras.models.load_model('ins+res.h5', custom_objects=custom_objects)

# Create VitisQuantizer object
quantizer = vitis_quantize.VitisQuantizer(model)

quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, 
                                           calib_steps=100, 
                                           calib_batch_size=10,
                                           ) 
quantized_model.save('quantized_model.h5')
