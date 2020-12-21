from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import keras
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint  

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from PIL import ImageFile    
from keras.layers.core import Activation
from sklearn.utils import shuffle

num_classes = 10
epochs = 30
learning_rate = 0.001
batch_size = 32
img_width, img_height, channels = 48, 48, 3

def load_dataset(path, shuffle):
    data = load_files(path,shuffle=shuffle)
    condition_files = np.array(data['filenames'])
    print(len(condition_files))
    condition_targets = np_utils.to_categorical(np.array(data['target']), num_classes)
    return condition_files, condition_targets

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img = np.float32(img)
    img = img/255
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def main():
    test_files, test_targets = load_dataset('testSet', False)

    print('There are %d test images.'% len(test_files))

    test_tensors = paths_to_tensor(test_files).astype('float32')/1

    from keras.models import load_model
    model = load_model('shallow.h5')

    condition_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    for c in condition_predictions:
        print("test number is: ", c)
    

if __name__ == "__main__":
    main()