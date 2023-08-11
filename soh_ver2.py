import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from google.colab import drive
drive.mount('/content/drive')
!pip install keras-octave-conv
import warnings
!pip install plot_keras_history
from plot_keras_history import show_history, plot_history
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from keras_octave_conv import OctaveConv2D
import pandas as pd
import tensorflow as tf
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import scipy.io
import numpy as np, h5py
from datetime import datetime
import glob, os
import json
import keras
%matplotlib inline
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D,LeakyReLU
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def getBatteryCapacity(Battery):
    cycle = []
    capacity = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'discharge':
            cycle.append(i)
            capacity.append(Bat['data']['Capacity'][0])
            i += 1
    return [cycle, capacity]
def getChargingValues(Battery, Index):
    Battery = Battery[Index]['data']
    index = []
    i = 1
    for iterator in Battery['Voltage_measured']:
        index.append(i)
        i += 1
    return [index, Battery['Voltage_measured'], Battery['Current_measured'], Battery['Temperature_measured'], Battery['Voltage_charge'], Battery['Time']]
def getDischargingValues(Battery, Index):
    Battery = Battery[Index]['data']
    index = []
    i = 1
    for iterator in Battery['Voltage_measured']:
        index.append(i)
        i += 1
    return [index, Battery['Voltage_measured'], Battery['Current_measured'], Battery['Temperature_measured'], Battery['Voltage_load'], Battery['Time']]
def getMaxDischargeTemp(Battery):
    cycle = []
    temp = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'discharge':
            cycle.append(i)
            temp.append(max(Bat['data']['Temperature_measured']))
            i += 1
    return [cycle, temp]
def getMaxChargeTemp(Battery, discharge_len):
    cycle = []
    temp = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'charge':
            cycle.append(i)
            temp.append(max(Bat['data']['Temperature_measured']))
            i += 1
    return [cycle[:discharge_len], temp[:discharge_len]]
def getDataframe(Battery):
    l = getBatteryCapacity(Battery)
    l1 = getMaxDischargeTemp(Battery)
    l2 = getMaxChargeTemp(Battery, len(l1[0]))
    data = {'cycle':l[0],'capacity':l[1], 'max_discharge_temp':l1[1], 'max_charge_temp':l2[1]}
    return pd.DataFrame(data)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def supervisedDataframeBuilder(Batterydataframe, scaler):
    values = Batterydataframe[['capacity']]
    scaled = scaler.fit_transform(values)
    data = series_to_supervised(scaled, 5, 1)
    data['cycle'] = data.index
    return data
def splitDataFrame(Dataframe, ratio):
    X = Dataframe[['cycle', 'var1(t-5)', 'var1(t-4)', 'var1(t-3)', 'var1(t-2)', 'var1(t-1)']]
    Y = Dataframe[['var1(t)']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = ratio, shuffle=False)
    return X_train, X_test, y_train, y_test
def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')
def rollingAverage(x_stuff, y_stuff):
    window_size = 10
    sigma=1.0
    avg = moving_average(y_stuff, window_size)
    avg_list = avg.tolist()
    residual = y_stuff - avg
    testing_std = residual.rolling(window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.iloc[window_size - 1]).round(3).iloc[:,0].tolist()
    rolling_std
    std = np.std(residual)
    lst=[]
    lst_index = 0
    lst_count = 0
    for i in y_stuff.index:
        if (y_stuff[i] > avg_list[lst_index] + (1.5 * rolling_std[lst_index])) | (y_stuff[i] < avg_list[lst_index] - (1.5 * rolling_std[lst_index])):
            lt=[i,x_stuff[i], y_stuff[i],avg_list[lst_index],rolling_std[lst_index]]
            lst.append(lt)
            lst_count+=1
        lst_index+=1
    lst_x = []
    lst_y = []
    for i in range (0,len(lst)):
        lst_x.append(lst[i][1])
        lst_y.append(lst[i][2])
    return lst_x, lst_y
def convert_to_time(hmm):
	return datetime(year=int(hmm[0]),month=int(hmm[1]),day=int(hmm[2]), hour=int(hmm[3]),minute=int(hmm[4]),second=int(hmm[5]))
def loadMat(matfile):
	data = scipy.io.loadmat(matfile)
	filename = matfile.split(".")[0]
	col = data[filename]
	col = col[0][0][0][0]
	size = col.shape[0]
	da = []
	for i in range(size):
		k=list(col[i][3][0].dtype.fields.keys())
		d1 = {}
		d2 = {}
		if str(col[i][0][0]) != 'impedance':
			for j in range(len(k)):
				t=col[i][3][0][0][j][0];
				l=[]
				for m in range(len(t)):
					l.append(t[m])
				d2[k[j]]=l
		d1['cycle']=str(col[i][0][0])
		d1['temp']=int(col[i][1][0])
		d1['time']=str(convert_to_time(col[i][2][0]))
		d1['data']=d2
		da.append(d1)
	return da
!cp /content/drive/MyDrive/Colab/B0005.mat ./ -R
!cp /content/drive/MyDrive/Colab/B0006.mat ./ -R
!cp /content/drive/MyDrive/Colab/B0007.mat ./ -R
!cp /content/drive/MyDrive/Colab/B0018.mat ./ -R
B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')
B0005_capacity = getBatteryCapacity(B0005)
B0006_capacity = getBatteryCapacity(B0006)
B0007_capacity = getBatteryCapacity(B0007)
B0018_capacity = getBatteryCapacity(B0018)
charging_labels = ['Voltage_measured','Current_measured','Temperature_measured']
B0005_charging = getChargingValues(B0005, 0)
B0006_charging = getChargingValues(B0006, 0)
B0007_charging = getChargingValues(B0007, 0)
B0018_charging = getChargingValues(B0018, 0)
B0005_discharging = getDischargingValues(B0005, 1)
B0006_discharging = getDischargingValues(B0006, 1)
B0007_discharging = getDischargingValues(B0007, 1)  
dfB0005 = getDataframe(B0005)
dfB0006 = getDataframe(B0006)
#sns.residplot(dfB0006['cycle'], dfB0006['capacity'])
dfB0007 = getDataframe(B0007)
#dfB0007.head(7)
X = dfB0007['cycle']
Y = dfB0007['capacity']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
lst_x, lst_y = rollingAverage(X_train, y_train)
d = {'X_train':X_train.values,'y_train':y_train.values}
d = pd.DataFrame(d)
d = d[~d['X_train'].isin(lst_x)]
X_train = d['X_train'];y_train = d['y_train'];X_train = X_train.astype("float32");X_test = X_test.astype("float32");y_train = y_train.astype("float32");y_test = y_test.astype("float32")
print(f"******************************************************************************************************************************************************8")
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
train_x = np.asarray(X_train).reshape(len(X_train), 1, 1, 1);
train_y = np.asarray(y_train).reshape(len(y_train), 1, 1);
test_x = np.asarray(X_test).reshape(len(X_test), 1, 1, 1);
test_y = np.asarray(y_test).reshape(len(y_test), 1, 1)
print(f"train_x.shape: {train_x.shape}, train_y.shape: {train_y.shape}, test_x.shape: {test_x.shape}, test_y.shape: {test_y.shape}");patience=400
x=train_x;y=train_y;validation_data=(test_x, test_y);epochs=1;batch_size=32;callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_mse", min_delta=1e-20000, patience=patience, verbose=1,)];
def build_model():
  input = keras.Input(shape=(1, 1, 1, ),batch_size=None, name='Input');x=input;filters=8
  high, low = OctaveConv2D(filters=filters, kernel_size=1, octave=1, ratio_out=0.5)(x)
  #high = keras.layers.BatchNormalization(renorm=True)(high);high = keras.layers.ReLU()(high)
  #low = keras.layers.BatchNormalization(renorm=True)(low);low = keras.layers.ReLU()(low)
  high = keras.layers.Conv2D(filters/2,(1, 1), kernel_initializer="he_normal", padding="same", name="Conv1")(high);
  #high = keras.layers.BatchNormalization(renorm=True)(high);high = keras.layers.ReLU()(high)
  x = Add()([high, low])
  new_shape = (x.get_shape()[1], x.get_shape()[2]*x.get_shape()[3])
  x = keras.layers.Reshape(target_shape=new_shape, name="reshape-before-recurrent-dense")(x)
  x = keras.layers.LSTM(16, name='recurrent_layer1')(x)
  x = keras.layers.Dense(1)(x)
  model = keras.models.Model(input, x, name='capacity-estimator')
  optimizer_name = tf.keras.optimizers.Adam()
  rmse=tf.keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)
  metrics = ['mse', 'mae', 'mape', 'msle', rmse]
  model.compile(loss='mse', optimizer=optimizer_name, metrics=metrics )
  return model
model = build_model()
#model.summary(line_length=110)
history = model.fit(x, y, batch_size=batch_size, epochs=1, validation_data=validation_data, callbacks=callbacks)
y_pred = model.predict(X.values.reshape(-1, 1, 1, 1))
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.plot(X, Y, color='green', label='Battery capacity data')
ax.plot(X, y_pred, color='red', label='Fitted model')
ax.set(xlabel='cycles', ylabel='soh', title='Discharging performance at 43°C')
ax.legend()
first_ratio = 96
second_ratio = 70
third_ratio = 40
forth_ratio = 4
model = build_model()
ratios = [first_ratio, second_ratio, third_ratio, forth_ratio]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
    lst_x, lst_y = rollingAverage(X_train, y_train)
    d = {'X_train':X_train.values,'y_train':y_train.values}
    d = pd.DataFrame(d)
    d = d[~d['X_train'].isin(lst_x)]
    X_train = d['X_train']
    y_train = d['y_train']
    X_train = d['X_train'];y_train = d['y_train'];X_train = X_train.astype("float32");X_test = X_test.astype("float32");y_train = y_train.astype("float32");y_test = y_test.astype("float32")
    X_train = X_train.values.reshape(-1, 1, 1, 1)
    X_test = X_test.values.reshape(-1, 1, 1, 1)
    y_train = y_train.values.reshape(-1, 1, 1)
    y_test = y_test.values.reshape(-1, 1, 1)
    epochs=1
    x=X_train
    y=y_train
    batch_size=16
    validation_data=(X_test, y_test)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_mse", min_delta=1e-20000, patience=patience, verbose=1,)]
    history = model.fit(x, y, batch_size=batch_size, epochs=1, validation_data=validation_data, callbacks=callbacks, verbose=0)
    if ratio == first_ratio:
        y_pred_first_ratio = model.predict(X.values.reshape(-1, 1, 1, 1))
    elif ratio == second_ratio:
        y_pred_second_ratio = model.predict(X.values.reshape(-1, 1, 1, 1))
    elif ratio == third_ratio:
        y_pred_third_ratio = model.predict(X.values.reshape(-1, 1, 1, 1))
    elif ratio == forth_ratio:
        y_pred_forth_ratio = model.predict(X.values.reshape(-1, 1, 1, 1))
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.plot(X, Y, color='black', label='Battery Capacity')
ax.plot(X, y_pred_first_ratio, color='red', label=f'Prediction with train size of {100-first_ratio}%')
ax.plot(X, y_pred_second_ratio, color='blue', label=f'Prediction with train size of {100-second_ratio}%')
ax.plot(X, y_pred_third_ratio, color='green', label=f'Prediction with train size of {100-third_ratio}%')
ax.plot(X, y_pred_forth_ratio, color='yellow', label=f'Prediction with train size of {100-forth_ratio}%')
ax.set(xlabel='cycles', ylabel='SOH', title='Model performance for Battery 07')
ax.legend()
#show_history(history)
#plot_history(history, path="interpolated.png", interpolate=True)
'''models = pd.DataFrame({'Train/Test': ['Train Set', 'Test Set'],
    'MSE': [min(history.history['mse']), min(history.history['val_mse'])], 'RMSE': [min(history.history['rmse']), min(history.history['val_rmse'])],
    'MAE': [min(history.history['mae']), min(history.history['val_mae'])], 'MAPE': [min(history.history['mape']), min(history.history['val_mape'])],
    'MSLE': [min(history.history['msle']), min(history.history['val_msle'])] })
models'''

#extracting I 
from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import os

def build_dictionaries(mess):

    discharge, charge, impedance = {}, {}, {}

    for i, element in enumerate(mess):

        step = element[0][0]

        if step == 'discharge':
            discharge[str(i)] = {}
            discharge[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1)*1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)        

            discharge[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            discharge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
            discharge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
            discharge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
            discharge[str(i)]["current_load"] = data[0][0][3][0].tolist()
            discharge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
            discharge[str(i)]["time"] = data[0][0][5][0].tolist()

        if step == 'charge':
            charge[str(i)] = {}
            charge[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1)*1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)        

            charge[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            charge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
            charge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
            charge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
            charge[str(i)]["current_load"] = data[0][0][3][0].tolist()
            charge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
            charge[str(i)]["time"] = data[0][0][5][0].tolist()

        if step == 'impedance':
            impedance[str(i)] = {}
            impedance[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1)*1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)        

            impedance[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            impedance[str(i)]["sense_current"] = {}
            impedance[str(i)]["battery_current"] = {}
            impedance[str(i)]["current_ratio"] = {}
            impedance[str(i)]["battery_impedance"] = {}
            impedance[str(i)]["rectified_impedance"] = {}

            impedance[str(i)]["sense_current"]["real"] = np.real(data[0][0][0][0]).tolist()
            impedance[str(i)]["sense_current"]["imag"] = np.imag(data[0][0][0][0]).tolist()

            impedance[str(i)]["battery_current"]["real"] = np.real(data[0][0][1][0]).tolist()
            impedance[str(i)]["battery_current"]["imag"] = np.imag(data[0][0][1][0]).tolist()

            impedance[str(i)]["current_ratio"]["real"] = np.real(data[0][0][2][0]).tolist()
            impedance[str(i)]["current_ratio"]["imag"] = np.imag(data[0][0][2][0]).tolist()

            impedance[str(i)]["battery_impedance"]["real"] = np.real(data[0][0][3]).tolist()
            impedance[str(i)]["battery_impedance"]["imag"] = np.imag(data[0][0][3]).tolist()

            impedance[str(i)]["rectified_impedance"]["real"] = np.real(data[0][0][4]).tolist()
            impedance[str(i)]["rectified_impedance"]["imag"] = np.imag(data[0][0][4]).tolist()

            impedance[str(i)]["re"] = float(data[0][0][5][0][0])
            impedance[str(i)]["rct"] = float(data[0][0][6][0][0])
            
    return discharge, charge, impedance

def save_json(dictionary, name):
    with open(name + '.json', 'w') as f:
        json.dump(dictionary, f, indent=4)

folder = './'
filenames = ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']

for filename in filenames:
    name = filename.split('.mat')[0]
    print(name)
    struct = loadmat(folder + '/' + filename)
    mess = struct[name][0][0][0][0]
    
    discharge, charge, impedance = build_dictionaries(mess)
    
    save_json(discharge, name + '_discharge')
    save_json(charge, name + '_charge')    
    save_json(impedance, name + '_impedance')  

with open('./B0005_discharge.json') as f_discharge05:    
    discharge_data05 = json.load(f_discharge05)
with open('./B0006_discharge.json') as f_discharge06:    
    discharge_data06 = json.load(f_discharge06)
with open('./B0007_discharge.json') as f_discharge07:    
    discharge_data07 = json.load(f_discharge07)
with open('./B0018_discharge.json') as f_discharge18:    
    discharge_data18 = json.load(f_discharge18)
times05=[]
currents05=[]
for cycle05 in discharge_data05.keys():
  times05.append(discharge_data05[cycle05]["time"])
  currents05.append(discharge_data05[cycle05]["current_battery"])
I_05_cycles=[]
for i in range(len(currents05)):
  I_05_cycles.append(np.asarray(currents05[i]))
I_05_cycles = np.asarray(I_05_cycles)
max05=0
for i in range(len(I_05_cycles)):
  max05=max(len(I_05_cycles[i]), max05)
I_05_cycles_pad=[]
for i in range(len(I_05_cycles)):
  I_05_cycles_pad.append(np.pad(I_05_cycles[i], (0, max05-len(I_05_cycles[i]))))
I_05_cycles_pad = np.asarray(I_05_cycles_pad)
I_05_times=[]
for i in range(len(times05)):
  I_05_times.append(np.asarray(times05[i]))
I_05_times = np.asarray(I_05_times)
max05=0
for i in range(len(I_05_times)):
  max05=max(len(I_05_times[i]), max05)
#I_05_times_pad=[]
#for i in range(len(I_05_times)):
#  I_05_times_pad.append(np.pad(I_05_times[i], (0, max05-len(I_05_times[i])), mode='edge'))
#I_05_times_pad = np.asarray(I_05_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B005")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
for i in range(len(I_05_times)):
  plt.plot(I_05_times[i], I_05_cycles[i])
  warnings.filterwarnings("ignore")
times06=[]
currents06=[]
for cycle06 in discharge_data06.keys():
  times06.append(discharge_data06[cycle06]["time"])
  currents06.append(discharge_data06[cycle06]["current_battery"])
I_06_cycles=[]
for i in range(len(currents06)):
  I_06_cycles.append(np.asarray(currents06[i]))
I_06_cycles = np.asarray(I_06_cycles)
max06=0
for i in range(len(I_06_cycles)):
  max06=max(len(I_06_cycles[i]), max06)
I_06_cycles_pad=[]
for i in range(len(I_06_cycles)):
  I_06_cycles_pad.append(np.pad(I_06_cycles[i], (0, max06-len(I_06_cycles[i]))))
I_06_cycles_pad = np.asarray(I_06_cycles_pad)
I_06_times=[]
for i in range(len(times06)):
  I_06_times.append(np.asarray(times06[i]))
I_06_times = np.asarray(I_06_times)
max06=0
for i in range(len(I_06_times)):
  max06=max(len(I_06_times[i]), max06)
#I_06_times_pad=[]
#for i in range(len(I_06_times)):
#  I_06_times_pad.append(np.pad(I_06_times[i], (0, max06-len(I_06_times[i])), mode='edge'))
#I_06_times_pad = np.asarray(I_06_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B006")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
for i in range(len(I_06_times)):
  plt.plot(I_06_times[i], I_06_cycles[i])
  warnings.filterwarnings("ignore")
times07=[]
currents07=[]
for cycle07 in discharge_data07.keys():
  times07.append(discharge_data07[cycle07]["time"])
  currents07.append(discharge_data07[cycle07]["current_battery"])
I_07_cycles=[]
for i in range(len(currents07)):
  I_07_cycles.append(np.asarray(currents07[i]))
I_07_cycles = np.asarray(I_07_cycles)
max07=0
for i in range(len(I_07_cycles)):
  max07=max(len(I_07_cycles[i]), max07)
I_07_cycles_pad=[]
for i in range(len(I_07_cycles)):
  I_07_cycles_pad.append(np.pad(I_07_cycles[i], (0, max07-len(I_07_cycles[i]))))
I_07_cycles_pad = np.asarray(I_07_cycles_pad)
I_07_times=[]
for i in range(len(times07)):
  I_07_times.append(np.asarray(times07[i]))
I_07_times = np.asarray(I_07_times)
max07=0
for i in range(len(I_07_times)):
  max07=max(len(I_07_times[i]), max07)
#I_07_times_pad=[]
#for i in range(len(I_07_times)):
#  I_07_times_pad.append(np.pad(I_07_times[i], (0, max07-len(I_07_times[i])), mode='edge'))
#I_07_times_pad = np.asarray(I_07_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B007")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
for i in range(len(I_07_times)):
  plt.plot(I_07_times[i], I_07_cycles[i])
  warnings.filterwarnings("ignore")
times18=[]
currents18=[]
for cycle18 in discharge_data18.keys():
  times18.append(discharge_data18[cycle18]["time"])
  currents18.append(discharge_data18[cycle18]["current_battery"])
I_18_cycles=[]
for i in range(len(currents18)):
  I_18_cycles.append(np.asarray(currents18[i]))
I_18_cycles = np.asarray(I_18_cycles)
max18=0
for i in range(len(I_18_cycles)):
  max18=max(len(I_18_cycles[i]), max18)
I_18_cycles_pad=[]
for i in range(len(I_18_cycles)):
  I_18_cycles_pad.append(np.pad(I_18_cycles[i], (0, max18-len(I_18_cycles[i]))))
I_18_cycles_pad = np.asarray(I_18_cycles_pad)
I_18_times=[]
for i in range(len(times18)):
  I_18_times.append(np.asarray(times18[i]))
I_18_times = np.asarray(I_18_times)
max18=0
for i in range(len(I_18_times)):
  max18=max(len(I_18_times[i]), max18)
#I_18_times_pad=[]
#for i in range(len(I_18_times)):
#  I_18_times_pad.append(np.pad(I_18_times[i], (0, max18-len(I_18_times[i])), mode='edge'))
#I_18_times_pad = np.asarray(I_18_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B018")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
for i in range(len(I_18_times)):
  plt.plot(I_18_times[i], I_18_cycles[i])
  warnings.filterwarnings("ignore")
times05=[]
voltages05=[]
for cycle05 in discharge_data05.keys():
  times05.append(discharge_data05[cycle05]["time"])
  voltages05.append(discharge_data05[cycle05]["voltage_battery"])
V_05_cycles=[]
for i in range(len(voltages05)):
  V_05_cycles.append(np.asarray(voltages05[i]))
V_05_cycles = np.asarray(V_05_cycles)
maxV05=0
for i in range(len(V_05_cycles)):
  maxV05=max(len(V_05_cycles[i]), maxV05)
V_05_cycles_pad=[]
for i in range(len(V_05_cycles)):
  V_05_cycles_pad.append(np.pad(V_05_cycles[i], (0, maxV05-len(V_05_cycles[i])), mode='edge'))
V_05_cycles_pad = np.asarray(V_05_cycles_pad)
V_05_times=[]
for i in range(len(times05)):
  V_05_times.append(np.asarray(times05[i]))
V_05_times = np.asarray(V_05_times)
maxV05=0
for i in range(len(V_05_times)):
  maxV05=max(len(V_05_times[i]), maxV05)
#V_05_times_pad=[]
#for i in range(len(V_05_times)):
#  V_05_times_pad.append(np.pad(V_05_times[i], (0, maxV05-len(V_05_times[i])), mode='edge'))
#V_05_times_pad = np.asarray(V_05_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B005")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (A)")
for i in range(len(V_05_times)):
  plt.plot(V_05_times[i], V_05_cycles[i])
  warnings.filterwarnings("ignore")
times06=[]
voltages06=[]
for cycle06 in discharge_data06.keys():
  times06.append(discharge_data06[cycle06]["time"])
  voltages06.append(discharge_data06[cycle06]["voltage_battery"])
V_06_cycles=[]
for i in range(len(voltages06)):
  V_06_cycles.append(np.asarray(voltages06[i]))
V_06_cycles = np.asarray(V_06_cycles)
maxV06=0
for i in range(len(V_06_cycles)):
  maxV06=max(len(V_06_cycles[i]), maxV06)
V_06_cycles_pad=[]
for i in range(len(V_06_cycles)):
  V_06_cycles_pad.append(np.pad(V_06_cycles[i], (0, maxV05-len(V_06_cycles[i])), mode='edge'))
V_06_cycles_pad = np.asarray(V_06_cycles_pad)
V_06_times=[]
for i in range(len(times06)):
  V_06_times.append(np.asarray(times06[i]))
V_06_times = np.asarray(V_06_times)
maxV06=0
for i in range(len(V_06_times)):
  maxV06=max(len(V_06_times[i]), maxV06)
#V_06_times_pad=[]
#for i in range(len(V_06_times)):
#  V_06_times_pad.append(np.pad(V_06_times[i], (0, maxV06-len(V_06_times[i])), mode='edge'))
#V_06_times_pad = np.asarray(V_06_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B006")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
for i in range(len(V_06_times)):
  plt.plot(V_06_times[i], V_06_cycles[i])
  warnings.filterwarnings("ignore")
times07=[]
voltages07=[]
for cycle07 in discharge_data07.keys():
  times07.append(discharge_data07[cycle07]["time"])
  voltages07.append(discharge_data07[cycle07]["voltage_battery"])
V_07_cycles=[]
for i in range(len(voltages07)):
  V_07_cycles.append(np.asarray(voltages07[i]))
V_07_cycles = np.asarray(V_07_cycles)
maxV07=0
for i in range(len(V_07_cycles)):
  maxV07=max(len(V_07_cycles[i]), maxV07)
V_07_cycles_pad=[]
for i in range(len(V_07_cycles)):
  V_07_cycles_pad.append(np.pad(V_07_cycles[i], (0, maxV07-len(V_07_cycles[i])), mode='edge'))
V_07_cycles_pad = np.asarray(V_07_cycles_pad)
V_07_times=[]
for i in range(len(times07)):
  V_07_times.append(np.asarray(times07[i]))
V_07_times = np.asarray(V_07_times)
maxV07=0
for i in range(len(V_07_times)):
  maxV07=max(len(V_07_times[i]), maxV07)
#V_07_times_pad=[]
#for i in range(len(V_07_times)):
#  V_07_times_pad.append(np.pad(V_07_times[i], (0, maxV07-len(V_07_times[i])), mode='edge'))
#V_07_times_pad = np.asarray(V_07_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B007")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
for i in range(len(V_07_times)):
  plt.plot(V_07_times[i], V_07_cycles[i])
  warnings.filterwarnings("ignore")
times18=[]
voltages18=[]
for cycle18 in discharge_data18.keys():
  times18.append(discharge_data18[cycle18]["time"])
  voltages18.append(discharge_data18[cycle18]["voltage_battery"])
V_18_cycles=[]
for i in range(len(voltages18)):
  V_18_cycles.append(np.asarray(voltages18[i]))
V_18_cycles = np.asarray(V_18_cycles)
maxV18=0
for i in range(len(V_18_cycles)):
  maxV18=max(len(V_18_cycles[i]), maxV18)
V_18_cycles_pad=[]
for i in range(len(V_18_cycles)):
  V_18_cycles_pad.append(np.pad(V_18_cycles[i], (0, maxV18-len(V_18_cycles[i])), mode='edge'))
V_18_cycles_pad = np.asarray(V_18_cycles_pad)
V_18_times=[]
for i in range(len(times18)):
  V_18_times.append(np.asarray(times18[i]))
V_18_times = np.asarray(V_18_times)
maxV18=0
for i in range(len(V_18_times)):
  maxV18=max(len(V_18_times[i]), maxV18)
#V_18_times_pad=[]
#for i in range(len(V_18_times)):
#  V_18_times_pad.append(np.pad(V_18_times[i], (0, maxV18-len(V_18_times[i])), mode='edge'))
#V_18_times_pad = np.asarray(V_18_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B018")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
for i in range(len(V_18_times)):
  plt.plot(V_18_times[i], V_18_cycles[i])
  warnings.filterwarnings("ignore")

times05=[]
temp05=[]
for cycle05 in discharge_data05.keys():
  times05.append(discharge_data05[cycle05]["time"])
  temp05.append(discharge_data05[cycle05]["temp_battery"])
T_05_cycles=[]
for i in range(len(temp05)):
  T_05_cycles.append(np.asarray(temp05[i]))
T_05_cycles = np.asarray(T_05_cycles)
maxT05=0
for i in range(len(T_05_cycles)):
  maxT05=max(len(T_05_cycles[i]), maxT05)
T_05_cycles_pad=[]
for i in range(len(T_05_cycles)):
  T_05_cycles_pad.append(np.pad(T_05_cycles[i], (0, maxT05-len(T_05_cycles[i])), mode='edge'))
T_05_cycles_pad = np.asarray(T_05_cycles_pad)
T_05_times=[]
for i in range(len(times05)):
  T_05_times.append(np.asarray(times05[i]))
T_05_times = np.asarray(T_05_times)
maxT05=0
for i in range(len(T_05_times)):
  maxT05=max(len(T_05_times[i]), maxT05)
#T_05_times_pad=[]
#for i in range(len(T_05_times)):
#  T_05_times_pad.append(np.pad(T_05_times[i], (0, maxV05-len(T_05_times[i])), mode='edge'))
#T_05_times_pad = np.asarray(T_05_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B005")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (A)")
for i in range(len(T_05_times)):
  plt.plot(T_05_times[i], T_05_cycles[i])
  warnings.filterwarnings("ignore")
times06=[]
temp06=[]
for cycle06 in discharge_data06.keys():
  times06.append(discharge_data06[cycle06]["time"])
  temp06.append(discharge_data06[cycle06]["temp_battery"])
T_06_cycles=[]
for i in range(len(temp06)):
  T_06_cycles.append(np.asarray(temp06[i]))
T_06_cycles = np.asarray(T_06_cycles)
maxT06=0
for i in range(len(T_06_cycles)):
  maxT06=max(len(T_06_cycles[i]), maxT06)
T_06_cycles_pad=[]
for i in range(len(T_06_cycles)):
  T_06_cycles_pad.append(np.pad(T_06_cycles[i], (0, maxT05-len(T_06_cycles[i])), mode='edge'))
T_06_cycles_pad = np.asarray(T_06_cycles_pad)
T_06_times=[]
for i in range(len(times06)):
  T_06_times.append(np.asarray(times06[i]))
T_06_times = np.asarray(T_06_times)
maxT06=0
for i in range(len(T_06_times)):
  maxT06=max(len(T_06_times[i]), maxT06)
#T_06_times_pad=[]
#for i in range(len(T_06_times)):
#  T_06_times_pad.append(np.pad(T_06_times[i], (0, maxT06-len(T_06_times[i])), mode='edge'))
#T_06_times_pad = np.asarray(T_06_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B006")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (C)")
for i in range(len(T_06_times)):
  plt.plot(T_06_times[i], T_06_cycles[i])
  warnings.filterwarnings("ignore")
times07=[]
temp07=[]
for cycle07 in discharge_data07.keys():
  times07.append(discharge_data07[cycle07]["time"])
  temp07.append(discharge_data07[cycle07]["temp_battery"])
T_07_cycles=[]
for i in range(len(temp07)):
  T_07_cycles.append(np.asarray(temp07[i]))
T_07_cycles = np.asarray(T_07_cycles)
maxT07=0
for i in range(len(T_07_cycles)):
  maxT07=max(len(T_07_cycles[i]), maxT07)
T_07_cycles_pad=[]
for i in range(len(T_07_cycles)):
  T_07_cycles_pad.append(np.pad(T_07_cycles[i], (0, maxT07-len(T_07_cycles[i])), mode='edge'))
T_07_cycles_pad = np.asarray(T_07_cycles_pad)
T_07_times=[]
for i in range(len(times07)):
  T_07_times.append(np.asarray(times07[i]))
T_07_times = np.asarray(T_07_times)
maxT07=0
for i in range(len(T_07_times)):
  maxT07=max(len(T_07_times[i]), maxT07)
#T_07_times_pad=[]
#for i in range(len(T_07_times)):
#  T_07_times_pad.append(np.pad(T_07_times[i], (0, maxV07-len(T_07_times[i])), mode='edge'))
#T_07_times_pad = np.asarray(T_07_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B007")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (A)")
for i in range(len(T_07_times)):
  plt.plot(T_07_times[i], T_07_cycles[i])
  warnings.filterwarnings("ignore")
times18=[]
temp18=[]
for cycle18 in discharge_data18.keys():
  times18.append(discharge_data18[cycle18]["time"])
  temp18.append(discharge_data18[cycle18]["temp_battery"])
T_18_cycles=[]
for i in range(len(temp18)):
  T_18_cycles.append(np.asarray(temp18[i]))
T_18_cycles = np.asarray(T_18_cycles)
maxT18=0
for i in range(len(T_18_cycles)):
  maxT18=max(len(T_18_cycles[i]), maxT18)
T_18_cycles_pad=[]
for i in range(len(T_18_cycles)):
  T_18_cycles_pad.append(np.pad(T_18_cycles[i], (0, maxT18-len(T_18_cycles[i])), mode='edge'))
T_18_cycles_pad = np.asarray(T_18_cycles_pad)
T_18_times=[]
for i in range(len(times18)):
  T_18_times.append(np.asarray(times18[i]))
T_18_times = np.asarray(T_18_times)
maxT18=0
for i in range(len(T_18_times)):
  maxT18=max(len(T_18_times[i]), maxT18)
#T_18_times_pad=[]
#for i in range(len(T_18_times)):
#  T_18_times_pad.append(np.pad(T_18_times[i], (0, maxT18-len(T_18_times[i])), mode='edge'))
#T_18_times_pad = np.asarray(T_18_times_pad)
fig = plt.figure(figsize = (7, 4))
ax = fig.add_axes([1, 1, 1, 1])
ax.set_title("Discharge plot for B018")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (C)")
for i in range(len(T_18_times)):
  plt.plot(T_18_times[i], T_18_cycles[i])
  warnings.filterwarnings("ignore")

dfB0005 = getDataframe(B0005)
dfB0006 = getDataframe(B0006)
dfB0007 = getDataframe(B0007)
dfB0018 = getDataframe(B0018)
cycle5 = dfB0005['cycle'].values;c_b5 = dfB0005['capacity'].values;cycle6 = dfB0006['cycle'].values;c_b6 = dfB0006['capacity'].values;
cycle7 = dfB0007['cycle'].values;c_b7 = dfB0007['capacity'].values;cycle18 = dfB0018['cycle'].values;c_b18 = dfB0018['capacity'].values;
#X_train, X_test, y_train, y_test = train_test_split(dfB0005['cycle'], dfB0005['capacity'], test_size=0.2, shuffle=False)
#preprocessing for data(check mse once without this and once with this)
'''
lst_x, lst_y = rollingAverage(X_train, y_train)
d = {'X_train':X_train.values,'y_train':y_train.values}
d = pd.DataFrame(d)
d = d[~d['X_train'].isin(lst_x)]
X_train = d['X_train'];y_train = d['y_train'];X_train = X_train.astype("float32");X_test = X_test.astype("float32");y_train = y_train.astype("float32");y_test = y_test.astype("float32")
'''
print(f"b5 cycle(c) and c:{cycle5.shape}, {c_b5.shape}_ b6 cycle(c) and c:{cycle6.shape}, {c_b6.shape}_ b7 cycle(c) and c:{cycle7.shape}, {c_b7.shape}_ b18 cycle(c) and c:{cycle18.shape}, {c_b18.shape}")

print(f"V_18_cycles_pad.shape(): {V_18_cycles_pad.shape}")
print(f"V_07_cycles_pad.shape(): {V_07_cycles_pad.shape}")
print(f"V_06_cycles_pad.shape(): {V_06_cycles_pad.shape}")
print(f"V_05_cycles_pad.shape(): {V_05_cycles_pad.shape}")
print(f"I_18_cycles_pad.shape(): {I_18_cycles_pad.shape}")
print(f"I_07_cycles_pad.shape(): {I_07_cycles_pad.shape}")
print(f"I_06_cycles_pad.shape(): {I_06_cycles_pad.shape}")
print(f"I_05_cycles_pad.shape(): {I_05_cycles_pad.shape}")

print(f"T_18_cycles_pad.shape(): {T_18_cycles_pad.shape}")
print(f"T_07_cycles_pad.shape(): {T_07_cycles_pad.shape}")
print(f"T_06_cycles_pad.shape(): {T_06_cycles_pad.shape}")
print(f"T_05_cycles_pad.shape(): {T_05_cycles_pad.shape}")
I_18_cycles_pad_pad=[]
for i in range(len(I_18_cycles_pad)):
  I_18_cycles_pad_pad.append(np.pad(I_18_cycles_pad[i], (0, 5), mode='constant'))
I_18_cycles_pad_pad = np.asarray(I_18_cycles_pad_pad)

V_18_cycles_pad_pad=[]
for i in range(len(V_18_cycles_pad)):
  V_18_cycles_pad_pad.append(np.pad(V_18_cycles_pad[i], (0, 5), mode='edge'))
V_18_cycles_pad_pad = np.asarray(V_18_cycles_pad_pad)

T_18_cycles_pad_pad=[]
for i in range(len(T_18_cycles_pad)):
  T_18_cycles_pad_pad.append(np.pad(T_18_cycles_pad[i], (0, 5), mode='edge'))
T_18_cycles_pad_pad = np.asarray(T_18_cycles_pad_pad)
I_05_06_07=np.concatenate((I_05_cycles_pad, I_06_cycles_pad, I_07_cycles_pad, ), axis=1);print(I_05_06_07.shape)
V_05_06_07=np.concatenate((V_05_cycles_pad, V_06_cycles_pad, V_07_cycles_pad, ), axis=1);print(V_05_06_07.shape)
T_05_06_07=np.concatenate((T_05_cycles_pad, T_06_cycles_pad, T_07_cycles_pad, ), axis=1);print(T_05_06_07.shape)
multi_18 = np.concatenate((I_18_cycles_pad_pad, V_18_cycles_pad_pad, T_18_cycles_pad_pad), axis=1); print(multi_18.shape)
multi_05_06_07 = np.concatenate((I_05_06_07, V_05_06_07, T_05_06_07), axis=0); print(multi_05_06_07.shape)
#c_18_multi = np.concatenate((c_b), axis=0)
c_05_06_07 = np.concatenate((c_b5, c_b6, c_b7), axis=0); print(c_05_06_07.shape)
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
multi_05_06_07 = NormalizeData(multi_05_06_07)
multi_18 = NormalizeData(multi_18)

multi_05_06_07=multi_05_06_07.reshape(504, 1113, 1)

model = Sequential()
model.add(LSTM(80,input_shape=(1113,1),return_sequences=False))
model.add(keras.layers.Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mse',optimizer ='adam',metrics=[['mse']])
#model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(multi_05_06_07,c_05_06_07,epochs=1,batch_size=64,validation_split=0.05,verbose=1, callbacks=[callback]);


predict=model.predict(multi_18.reshape(132, 1113, 1))

plt.plot(c_05_06_07)
plt.plot(c_b18)
plt.plot(predict)
scalar_x = MinMaxScaler(feature_range=(0, 1));X=scalar_x.fit_transform(V_05_cycles_pad);print(X.shape)
c_b5=c_b5.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));Y=scalar_y.fit_transform(c_b5);print(Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False);
x_train = x_train.reshape(len(x_train), x_train.shape[1], 1);print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
x_test = x_test.reshape(len(x_test), x_test.shape[1], 1);print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(LSTM(16))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#model.summary()
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)]
history=model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=2, validation_data=(x_test, y_test), callbacks=callbacks, shuffle=False)
y_predict = model.predict(X.reshape(-1, X.shape[1], 1))
fig, ax = plt.subplots(1, figsize=(12, 8))
i=170;ax.plot(np.arange(i), 0.2*np.ones((i, 1)),'k--',linewidth = 2)
ax.plot(Y, color='black',label='Actual Capacity')
#plt.plot(y_test)
ax.plot(y_predict, color='red',label='Predicted Capacity')
ax.set(xlabel='Discharge Cycles', ylabel='Capacity(Ah)')
ax.set_xlim([0,170])
ax.legend()
ax.set(title = 'LSTM')
warnings.filterwarnings("ignore")
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();
print(V_05_cycles_pad.shape),print(I_05_cycles_pad.shape),print(T_05_cycles_pad.shape);
scalar_x = MinMaxScaler(feature_range=(0, 1));
V_05_cycles_pad = scalar_x.fit_transform(V_05_cycles_pad)
I_05_cycles_pad = scalar_x.fit_transform(I_05_cycles_pad)
T_05_cycles_pad = scalar_x.fit_transform(T_05_cycles_pad)
vit_5=np.concatenate((V_05_cycles_pad.reshape(-1, V_05_cycles_pad.shape[1], 1),
                     I_05_cycles_pad.reshape(-1, I_05_cycles_pad.shape[1], 1),
                     T_05_cycles_pad.reshape(-1, T_05_cycles_pad.shape[1], 1)), axis=2);X=vit_5;print(X.shape)
c_b5=c_b5.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));Y=scalar_y.fit_transform(c_b5);print(Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False);
x_train.shape, x_test.shape, y_train.shape, y_test.shape
#x_train = x_train.reshape(len(x_train), x_train.shape[1], 1);print(f"x_train.shape: {x_train.shape}")
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(LSTM(16))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#model.summary()
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)]
history=model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=2, validation_data=(x_test, y_test), callbacks=callbacks, shuffle=False)
y_predict = model.predict(X)
fig, ax = plt.subplots(1, figsize=(12, 8))
i=170;ax.plot(np.arange(i), 0.2*np.ones((i, 1)),'k--',linewidth = 2)
ax.plot(Y, color='black',label='Actual Capacity')
#plt.plot(y_test)
ax.plot(y_predict, color='red',label='Predicted Capacity')
ax.set(xlabel='Discharge Cycles', ylabel='Capacity(Ah)')
ax.set_xlim([0,170])
ax.legend()
ax.set(title = 'LSTM')
warnings.filterwarnings("ignore")
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();










I_18_cycles_pad_pad=[]
for i in range(len(I_18_cycles_pad)):
  I_18_cycles_pad_pad.append(np.pad(I_18_cycles_pad[i], (0, 5), mode='constant'))
for i in range(36):
  I_18_cycles_pad_pad.append(I_18_cycles_pad_pad[131])
I_18_cycles_pad_pad = np.asarray(I_18_cycles_pad_pad);print(f"I_18_cycles_pad_pad: {I_18_cycles_pad_pad.shape}")

V_18_cycles_pad_pad=[]
for i in range(len(V_18_cycles_pad)):
  V_18_cycles_pad_pad.append(np.pad(V_18_cycles_pad[i], (0, 5), mode='edge'))
for i in range(36):
  V_18_cycles_pad_pad.append(V_18_cycles_pad_pad[131])
V_18_cycles_pad_pad = np.asarray(V_18_cycles_pad_pad); print(f"V_18_cycles_pad_pad: {V_18_cycles_pad_pad.shape}")

T_18_cycles_pad_pad=[]
for i in range(len(T_18_cycles_pad)):
  T_18_cycles_pad_pad.append(np.pad(T_18_cycles_pad[i], (0, 5), mode='edge'))
for i in range(36):
  T_18_cycles_pad_pad.append(T_18_cycles_pad_pad[131])
T_18_cycles_pad_pad = np.asarray(T_18_cycles_pad_pad); print(f"T_18_cycles_pad_pad: {T_18_cycles_pad_pad.shape}")
c_b18 = dfB0018['capacity'].values;c_b18=c_b18.tolist()
for i in range(36):
  c_b18.append(c_b18[131])
c_b18 = np.asarray(c_b18);c_b18=c_b18.reshape(-1, 1);print(f"c_b18: {c_b18.shape}")
scalar_x = MinMaxScaler(feature_range=(0, 1));
V_05_cycles_pad = scalar_x.fit_transform(V_05_cycles_pad)
I_05_cycles_pad = scalar_x.fit_transform(I_05_cycles_pad)
T_05_cycles_pad = scalar_x.fit_transform(T_05_cycles_pad)
vit_5=np.concatenate((V_05_cycles_pad.reshape(-1, V_05_cycles_pad.shape[1], 1),
                     I_05_cycles_pad.reshape(-1, I_05_cycles_pad.shape[1], 1),
                     T_05_cycles_pad.reshape(-1, T_05_cycles_pad.shape[1], 1)), axis=2);print(f"vit5: {vit_5.shape}")
scalar_x = MinMaxScaler(feature_range=(0, 1));
V_06_cycles_pad = scalar_x.fit_transform(V_06_cycles_pad)
I_06_cycles_pad = scalar_x.fit_transform(I_06_cycles_pad)
T_06_cycles_pad = scalar_x.fit_transform(T_06_cycles_pad)
vit_6=np.concatenate((V_06_cycles_pad.reshape(-1, V_06_cycles_pad.shape[1], 1),
                     I_06_cycles_pad.reshape(-1, I_06_cycles_pad.shape[1], 1),
                     T_06_cycles_pad.reshape(-1, T_06_cycles_pad.shape[1], 1)), axis=2);print(f"vit6: {vit_6.shape}")
scalar_x = MinMaxScaler(feature_range=(0, 1));
V_07_cycles_pad = scalar_x.fit_transform(V_07_cycles_pad)
I_07_cycles_pad = scalar_x.fit_transform(I_07_cycles_pad)
T_07_cycles_pad = scalar_x.fit_transform(T_07_cycles_pad)
vit_7=np.concatenate((V_07_cycles_pad.reshape(-1, V_07_cycles_pad.shape[1], 1),
                     I_07_cycles_pad.reshape(-1, I_07_cycles_pad.shape[1], 1),
                     T_07_cycles_pad.reshape(-1, T_07_cycles_pad.shape[1], 1)), axis=2);print(f"vit7: {vit_7.shape}")
scalar_x = MinMaxScaler(feature_range=(0, 1));
V_18_cycles_pad_pad = scalar_x.fit_transform(V_18_cycles_pad_pad)
I_18_cycles_pad_pad = scalar_x.fit_transform(I_18_cycles_pad_pad)
T_18_cycles_pad_pad = scalar_x.fit_transform(T_18_cycles_pad_pad)
vit_18=np.concatenate((V_18_cycles_pad_pad.reshape(-1, V_18_cycles_pad_pad.shape[1], 1),
                     I_18_cycles_pad_pad.reshape(-1, I_18_cycles_pad_pad.shape[1], 1),
                     T_18_cycles_pad_pad.reshape(-1, T_18_cycles_pad_pad.shape[1], 1)), axis=2);print(f"vit18: {vit_18.shape}")
vit_5_6_7=np.concatenate((vit_5, vit_6, vit_7, vit_18), axis=0);X=vit_5_6_7;print(f"X: {X.shape}")
c_b5=c_b5.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));Y5=scalar_y.fit_transform(c_b5);print(f"Y5.shape: {Y5.shape}")
c_b6=c_b6.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));Y6=scalar_y.fit_transform(c_b6);print(f"Y6.shape: {Y6.shape}")
c_b7=c_b7.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));Y7=scalar_y.fit_transform(c_b7);print(f"Y7.shape: {Y7.shape}")
c_b18=c_b18.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));Y18=scalar_y.fit_transform(c_b18);print(f"Y18.shape: {Y18.shape}")
c_b567_18 = np.concatenate((Y5, Y6, Y7, Y18), axis=0);Y=c_b567_18;print(f"Y.shape: {Y.shape}")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle=False);
x_train.shape, x_test.shape, y_train.shape, y_test.shape











scalar_x = MinMaxScaler(feature_range=(0, 1));
V_05_cycles_pad = scalar_x.fit_transform(V_05_cycles_pad)
c_b5=c_b5.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));c_b5=scalar_y.fit_transform(c_b5);
X5 = V_05_cycles_pad.reshape(len(V_05_cycles_pad), V_05_cycles_pad.shape[1], 1); print(f"X5.shape: {X5.shape}")
Y5 = c_b5;print(f"Y5.shape: {Y5.shape}")


scalar_x = MinMaxScaler(feature_range=(0, 1));
V_06_cycles_pad = scalar_x.fit_transform(V_06_cycles_pad)
c_b6=c_b6.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));c_b6=scalar_y.fit_transform(c_b6);
X6 = V_06_cycles_pad.reshape(len(V_06_cycles_pad), V_06_cycles_pad.shape[1], 1);print(f"X6.shape: {X6.shape}")
Y6 = c_b6;print(f"Y6.shape: {Y6.shape}")



scalar_x = MinMaxScaler(feature_range=(0, 1));
V_07_cycles_pad = scalar_x.fit_transform(V_07_cycles_pad)
c_b7=c_b7.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));c_b7=scalar_y.fit_transform(c_b7);
X7 = V_07_cycles_pad.reshape(len(V_07_cycles_pad), V_07_cycles_pad.shape[1], 1);print(f"X7.shape: {X7.shape}")
Y7 = c_b7;print(f"Y7.shape: {Y7.shape}")




scalar_x = MinMaxScaler(feature_range=(0, 1));
V_18_cycles_pad_pad=[]
for i in range(len(V_18_cycles_pad)):
  V_18_cycles_pad_pad.append(np.pad(V_18_cycles_pad[i], (0, 5), mode='edge'))
for i in range(36):
  V_18_cycles_pad_pad.append(V_18_cycles_pad_pad[131])
V_18_cycles_pad_pad = np.asarray(V_18_cycles_pad_pad);
V_18_cycles_pad_pad = scalar_x.fit_transform(V_18_cycles_pad_pad)
X18=V_18_cycles_pad_pad.reshape(len(V_18_cycles_pad_pad), V_18_cycles_pad_pad.shape[1], 1);print(f"X18: {X18.shape}")
c_b18=c_b18.reshape(-1, 1);scalar_y = MinMaxScaler(feature_range=(0, 1));c_b18=scalar_y.fit_transform(c_b18);Y=c_b18;print(f"Y18.shape: {Y18.shape}")




V_5618=np.concatenate((X5, X6, X7, X18), axis=2); print(V_5618.shape)
c_5618=np.concatenate((Y5, Y6, Y7, Y18), axis=1); print(c_5618.shape)






x_train, x_test, y_train, y_test = train_test_split(V_5618, c_5618, test_size=0.2, shuffle=False);




model = Sequential()
model.add(LSTM(40, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(LSTM(16))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(4))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#model.summary()
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)]



history=model.fit(x_train, y_train, epochs=800, batch_size=32, verbose=2, validation_data=(x_test, y_test), callbacks=callbacks, shuffle=False)





q=168*3
p=168*4
fig, ax = plt.subplots(1, figsize=(12, 8))
#i=170;ax.plot(np.arange(i), 0.2*np.ones((i, 1)),'k--',linewidth = 2)
ax.plot(c_b5, color='black',label='Actual Capacity')
#plt.plot(y_test)
ax.plot(scalar_y.inverse_transform(y_predict[:,:1]), color='red',label='Predicted Capacity')
ax.set(xlabel='Discharge Cycles', ylabel='Capacity(Ah)')
ax.set_xlim([0,125])
ax.legend()
ax.set(title = 'LSTM')
warnings.filterwarnings("ignore")
