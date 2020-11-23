import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.legend()
  plt.show()

length = 20

f = open("born_for_age.txt", encoding="UTF8")
lines = [line.split() for line in f.readlines()]

output_data = []
for line in lines[1:]:
    for data in line[3:11]:
        output_data.append(float(data.replace(",", "")))
output_data = np.array(output_data)
output_data = (output_data - np.mean(output_data))/np.std(output_data)

input_data = []
for i in range(length):
    input_data.extend([17, 22, 27, 32, 37, 42, 47, 52])
input_data = np.array(input_data)
input_data = (input_data - np.mean(input_data))/np.std(input_data)

# print(input_data)
# print(output_data)
print(input_data.shape)
print(output_data.shape)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1, )))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="mse",
    metrics=["mse"]
)

print(model.summary())

history = model.fit(
    input_data,
    output_data,
    epochs=3000,
    validation_split=0.2
)
# plot_history(history)

plt.plot(input_data[:8], output_data[:8])
x = list(np.arange(input_data[0], input_data[7], 0.05))
y = [float(model.predict([i])) for i in x]
plt.plot(x, y)
plt.show()