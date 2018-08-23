from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing import image
from keras import Input, Model
from keras.layers import Conv2D, Dense, MaxPool2D, Add, Flatten, Concatenate, Dropout
from keras.models import load_model
import matplotlib.pyplot as plt


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
test_generator = train_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

input_ = Input(shape=(150,150,3))
conv_1 = Conv2D(8, (7, 7), activation='relu')(input_)
max_1 = MaxPool2D((2,2))(conv_1)
conv_2 = Conv2D(16, (5, 5), activation='relu')(max_1)
max_2 = MaxPool2D((2,2))(conv_2)
conv_3 = Conv2D(32, (3,3), activation='relu')(max_2)
max_3 = MaxPool2D((2,2))(conv_3)
conv_4 = Conv2D(64, (3,3), activation = 'relu')(max_3)
max_4 = MaxPool2D((3,3))(conv_4)
conv_5 = Conv2D(128, (4,4), activation = 'relu')(max_4) 
flatten_ = Flatten()(conv_5)
dense_1 = Dense(256, activation = 'relu')(flatten_)
dropout_1 = Dropout(0.1)(dense_1)
dense_2 = Dense(128, activation = 'relu')(dense_1)
dropout_2 = Dropout(0.2)(dense_2)
out = Dense(1, activation = 'tanh')(dropout_2)



model = Model(inputs=input_, outputs=out)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,  
    epochs = 30,
    validation_data=test_generator,
    validation_steps=50
)

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
