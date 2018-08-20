from keras.models import load_model
import matplotlib.image as mpimg
import numpy as np
from keras.utils import np_utils

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

model = load_model('my_model.h5')


img = mpimg.imread('otte.png')
np.shape(img)
gray = rgb2gray(img)
#print(gray)
x = np.reshape(gray, (1,784))

print(model.predict(x))