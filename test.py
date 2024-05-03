import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import numpy as np, tensorflow as tf, cv2, pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score
from train import create_dir, load_dataset
from glob import glob

global image_h, image_w, num_classes, classes, rgb_codes


if __name__=="__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results")
    image_h=128
    image_w=128
    num_classes=11

    dataset_path= "./data"
    model_path= os.path.join('files', 'model.keras')

    rgb_codes=[
        [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
        [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
        [102, 0, 51], [255, 204, 255], [255, 0, 102]
    ]
    classes=["background", "skin", "left_eyebrow", "right_eyebrow", "left eye",
             "right eye", "nose", "upper lip", "inner_mouth", "lower lip", "hair"]
    

    (train_x, train_y),(val_x, val_y),(test_x, test_y)= load_dataset(dataset_path)
    model= tf.keras.models.load_model(model_path)

    SCORE=[]

    for x, y in tqdm(zip(test_x, test_y), total= len(test_x)):
        print(x, y)
        
