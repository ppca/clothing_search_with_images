"""This helper file contains helper functions for clothing project"""

import numpy as np
import os
from keras.utils import np_utils
from keras.preprocessing import image                  
from tqdm import tqdm


def read_label_txt(label_txt):
    img_label_map = {}
    cnt = 0
    with open(label_txt, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            cnt += 1
            if cnt >= 3:
                contents = line[:-1].split(' ')
                file = contents[0]
                label = int(contents[-1])
                img_label_map[file] = label
    return img_label_map


def load_dataset(image_folder_path, label_txt):
    """Return the names of clothing categories, file paths and their corresponding class.
       :params
       image_folder_path: the folder where all images sit
       :return
       cloth_names: list[str]
       cloth_files: list[st]
       cloth_targets: list[int]
    """
    images = []
    targets = []

    img_label_map = read_label_txt(label_txt)

    for sub_folder in os.listdir(image_folder_path):
        if os.path.isdir(image_folder_path + sub_folder + '/'):
            for file in os.listdir(image_folder_path + sub_folder + '/'):
                img_name = 'img/'+sub_folder+'/'+file
                if img_name in img_label_map:
                    images.append(image_folder_path + sub_folder + '/'+file)
                    targets.append(img_label_map[img_name])
    cloth_files = np.array(images)
    cloth_targets = np_utils.to_categorical(np.array(targets), np.max(targets))
    return cloth_files, cloth_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def train_val_test_split(cloth_files, cloth_targets, ratio= 0.8):
    """return training, validation and test data
       : param
       cloth_files: list of all clothing files
       cloth_targets: list of all clothing targets
       ratio: train test split ratio
       :return
       train_files, train_targets, val_iles, val_targets, test_files, test_targets
    """
    train_files = []
    val_files = []
    test_files = []
    train_targets = []
    val_targets = []
    test_targets = []
    chances = np.random.sample(size = len(cloth_files))
    for idx, p in enumerate(chances):
        if p > ratio:
            test_files.append(cloth_files[idx])
            test_targets.append(cloth_targets[idx])
        elif p > ratio**2:
            val_files.append(cloth_files[idx])
            val_targets.append(cloth_targets[idx])
        else:
            train_files.append(cloth_files[idx])
            train_targets.append(cloth_targets[idx])
    return train_files, train_targets, val_files, val_targets, test_files, test_targets




