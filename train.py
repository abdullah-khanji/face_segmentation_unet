import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import numpy as np, pandas as pd, cv2, scipy.io, tensorflow as tf
from glob import glob
from unet import build_unet

global image_h, image_w, num_classes, classes, rgb_codes

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x= sorted(glob(os.path.join(path, 'train', 'images', '*.jpg')))
    train_y= sorted(glob(os.path.join(path, "train", "labels", "*.png")))

    valid_x= sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y= sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    test_x= sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y= sorted(glob(os.path.join(path, "test", "labels", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image_mask(x, y):
    x= cv2.imread(x, cv2.IMREAD_COLOR)
    x= cv2.resize(x, (image_w, image_h))
    x= x/255.0
    x= x.astype(np.float32)

    #mask
    y= cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y= cv2.resize(y, (image_w, image_h))
    y= y.astype(np.int32)

    return x, y

def preprocess(x, y):
    def f(x, y):
        x= x.decode()
        y= y.decode()
        return read_image_mask(x, y)
    
    image, mask= tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask= tf.one_hot(mask, num_classes)

    image.set_shape([image_h, image_w, 3])
    image.set_shape([image_h, image_w, num_classes])

    return image, mask

def tf_dataset(X, Y, batch=8):
    ds= tf.data.Dataset.from_tensor_slices((X, Y))
    ds= ds.shuffle(buffer_size=5000).map(preprocess)
    ds= ds.batch(batch).prefetch(2)
    return ds


if __name__=='__main__':
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")
    image_h= 256
    image_w= 256
    num_classes=11

    input_shape= (image_h, image_w, 3)
    batch_size=8
    lr=1e-4 #0.0001
    num_epochs=100

    dataset_path= './data'
    model_path=os.path.join('files', "model.keras")
    csv_path= os.path.join("files","log.csv")




    (train_x, train_y), (valid_x, valid_y), (test_x, test_y)= load_dataset(dataset_path)

    print("train: {}- {}".format(len(train_x),len(train_y)))
    print("valid: {}- {}".format(len(valid_x),len(valid_y)))
    print("test: {}- {}".format(len(test_x),len(test_y)))

    train_dataset= tf_dataset(train_x, train_y, batch_size)
    valid_dataset= tf_dataset(valid_x, valid_y, batch_size)
    test_dataset= tf_dataset(test_x, test_y, batch_size)

    model = build_unet((128, 128, 3))
    model.compile()