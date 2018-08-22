from keras.layers import Conv2D, Dense, MaxPool2D, Add, Flatten, Concatenate, Dropout
from keras.datasets import mnist
from keras import Input, Model
from keras.utils import np_utils, plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

# Normalizing
x_train = x_train / 255
x_test = x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

input_1 = Input(shape=(28, 28, 1))

conv2d_1 = Conv2D(16, (5, 5), input_shape=(28, 28, 1), activation='relu')(input_1)
conv2d_2 = Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu')(input_1)

max_pooling2d_1 = MaxPool2D((13, 13))(conv2d_1)
max_pooling2d_2 = MaxPool2D((12, 12))(conv2d_2)

add_1 = Add()([max_pooling2d_1, max_pooling2d_2])

conv2d_3 = Conv2D(48, (2, 2), activation='relu')(add_1)

flatten_1 = Flatten()(conv2d_3)

dense_1 = Dense(64, activation='relu')(flatten_1)
dense_2 = Dense(24, activation='relu')(dense_1)

concatenate_1 = Concatenate()([dense_2, flatten_1])

dense_3 = Dense(10, activation='softmax')(concatenate_1)

model = Model(inputs=input_1, outputs=dense_3)

# plot_model( model , to_file='model.png')

callback_list = [
#    ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
]

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=128,
                    validation_data=(x_test, y_test),
                    verbose=2,
                    callbacks=callback_list)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)


# ’bo’ is for blue dot, ‘b’ is for solid blue line
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('my_model.h5')
