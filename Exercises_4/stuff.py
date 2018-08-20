from keras.models import load_model
import matplotlib.image as mpimg
import numpy as np
from keras.utils import np_utils

model = load_model('my_model.h5')

model.summary()
