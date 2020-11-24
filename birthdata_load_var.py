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
input_data_year = input_data[:, 0]
input_data_age = input_data[:, 1]

input_data_year_mean = np.mean(input_data_year)
input_data_year_std = np.std(input_data_year)
input_data_year = (input_data_year - input_data_year_mean)/input_data_year_std

input_data_age_mean = np.mean(input_data_age)
input_data_age_std = np.std(input_data_age)
input_data_age = (input_data_age - input_data_age_mean)/input_data_age_std

input_data = np.array([input_data_year, input_data_age]).T

output_data = np.array(output_data)
output_data_mean = np.mean(output_data)
output_data_std = np.std(output_data)
output_data = (output_data - output_data_mean)/output_data_std

print(input_data.shape)
print(output_data.shape)

model = keras.models.load_model("birthdata_model")

X = np.arange(input_data[0, 1], input_data[9, 1]+1, 0.05)
Y = [float(model.predict([[input_data[0, 0], x]])) for x in X]

plt.plot(np.multiply(input_data[:10, 1], input_data_age_std) + input_data_age_mean, np.multiply(output_data[:10], output_data_std) + output_data_mean, label="target")
plt.plot(np.multiply(X, input_data_age_std) + input_data_age_mean, np.multiply(Y, output_data_std) + output_data_mean, label="pred.")

plt.legend()
plt.show()

print(np.multiply(X, input_data_age_std) + input_data_age_mean)
print(np.multiply(Y, output_data_std) + output_data_mean)