from keras.datasets import imdb
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Input, Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

max_features = 10000
max_len = 128
out_dimensions = 16

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(train_data, maxlen=max_len)
x_test = pad_sequences(test_data, maxlen=max_len)

model = Sequential()

model.add(Embedding(max_features, out_dimensions, input_length=max_len))
model.add(LSTM(out_dimensions, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train,
                    train_labels,
                    epochs=20,
                    batch_size=128,
                    validation_data=(x_test, test_labels),
                    verbose=1)

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

model.save('my_model_rnn.h5')
