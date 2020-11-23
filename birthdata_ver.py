import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import csv


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()
    plt.show()


length = 20

f = open("birthdata.csv", encoding="UTF8")
data = list(csv.reader(f))

input_data = []
output_data = []
for line in data[2:]:
    if line[1] != 'Total' and line[1] != 'Unknown' and not ('+' in line[1]) :
        age_temp = line[1].replace("-", "").split()
        age_avg = (float(age_temp[0]) + float(age_temp[1]))/2
        input_data.append([float(line[0]), age_avg])
        output_data.append(float(line[2]))

input_data = np.array(input_data)
# input_data = (input_data - np.mean(input_data))/np.std(input_data)

output_data = np.array(output_data)
output_data = (output_data - np.mean(output_data))/np.std(output_data)

print(input_data.shape)
print(output_data.shape)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(2, )))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
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
    epochs=10,
    validation_split=0.2
)
# plot_history(history)

plt.plot(input_data[:10, 1], output_data[:10])

X = np.arange(input_data[0, 1], input_data[9, 1], 1)
Y = [float(model.predict([[input_data[0, 0], x]])) for x in X]
plt.plot(X, Y)
plt.show()