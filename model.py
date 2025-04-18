<<<<<<< HEAD
import speasy as spz
from speasy import amda 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

=======
import os
os.environ['KERAS_BACKEND'] = 'torch'
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
#choose kernel (keras-env) when run on Macbook Pro 
<<<<<<< HEAD
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load the TensorBoard notebook extension
import tensorflow as tf
from datetime import datetime
import calendar
=======
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
<<<<<<< HEAD
import os
import random

def test_model(input_features,output_targets):
    model = Sequential()
    model.add(LSTM(128, input_dim=len(input_features), 
                kernel_initializer='he_uniform', 
                return_sequences=True,
                activation='relu'))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(output_targets)))
    model.compile(optimizer= optimizers.Adam(learning_rate=0.001), loss='mse')#, metrics=['mean_absolute_error'])
    return model

def first_dnn(input_features,output_targets):
    model = Sequential([
        Dense(len(input_features), activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(output_targets))
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
=======
from datetime import datetime
import calendar
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import toml
import h5py

def first_dnn(input_features,output_targets):
    model = keras.Sequential([
        layers.Dense(len(input_features), activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(output_targets))
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=keras.optimizers.Adam(0.001))
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32

    return model

def E_mhd(df):
    '''
    [B] = nT
    [u] = km/s
    [E] = µV/m => mV/m
    '''
    dg = pd.DataFrame(index=df.index,columns=['ex','ey','ez'])
    dg['ex']=(df['by']*df['uz']-df['uy']*df['bz'])/1e3
<<<<<<< HEAD
    dg['ey']=-(df['bz']*df['ux']-df['uz']*df['bx'])/1e3 #MINUS SIGN TO FIT MEASUREMENTS IN MMS1 MAYBE THERE IS A ISSUE WITH THE MEASUREMENT TOOL IN MMS1
=======
    dg['ey']=(df['bz']*df['ux']-df['uz']*df['bx'])/1e3 #MINUS SIGN TO FIT MEASUREMENTS IN MMS1 MAYBE THERE IS A ISSUE WITH THE MEASUREMENT TOOL IN MMS1
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
    dg['ez']=(df['bx']*df['uy']-df['ux']*df['by'])/1e3
    return dg

def E_hall(df):
    '''
    [B] = nT
    [j] = A/m^2
    [n] = cm^-3
    [E] = *1e-15 V/m => 1e-12 mV/m
    '''
    dg = pd.DataFrame(index=df.index,columns=['ex','ey','ez'])
    dg['ex']= ((df['jy']*df['bz']-df['by']*df['jz'])/(e*df['e_density']))/1e12
    dg['ey']= ((df['jz']*df['bx']-df['bz']*df['jx'])/(e*df['e_density']))/1e12
    dg['ez']= ((df['jx']*df['by']-df['bx']*df['jy'])/(e*df['e_density']))/1e12
    return dg

def lstm_data_transform(x_data, y_data, num_steps=5):
    """ Changes data to the format for LSTM training
for sliding window approach """    # Prepare the list for the transformed data
    X, y = list(), list()    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix]        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)    
    return x_array, y_array

e = 1.602176634e-19
m_i = 1.67262192369e-27
m_e = 9.1093837015e-31
if __name__ == "__main__":
    ## Global interval - all the data will be inside this range
<<<<<<< HEAD
    sat = 'mms1'
    t1 = datetime(2015,9,7,0,0,0)
    t2 = datetime(2015,9,30,0,0,0)
    f_train, f_valid, f_test = 0.80, 0.1, 0.1
    seed = 1
    PINNS = False #Do we physics inform the model ? i.e. giving cross products

=======
    with open('./model_config.toml','r') as f:
        config = toml.load(f)
        seed = config['stat']['seed']
        t1 = config['data']['t1']
        t2 = config['data']['t2']
        sat = config['data']['sat']
        density_threshold = config['data']['density_threshold']
        name = config['model']['name']
        f_train, f_valid, f_test = config['model']['f_train'], config['model']['f_valid'], config['model']['f_test']
        shuffle = config['model']['shuffle']
        PINNS = config['model']['PINNS']
        epochs = config['model']['epochs']
        patience = config['model']['patience']
        data_path = config['paths']['data']
        saved_models_path = config['paths']['saved_models']
        log_path = config['paths']['logs']
    #with open(f'{saved_models_path}/{name}/model.toml','w') as f:
    #    toml.dump(config,f)
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
<<<<<<< HEAD
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    ## Loading data
    store = pd.HDFStore('./data.hdf5','r')
    bursts = store.keys()
    store.close()
    del store

    bursts = [datetime.strptime(burst, '/mms1_%Y_%m_%dT%H_%M_%S') for burst in bursts]
    bursts = pd.DataFrame(bursts)

    bursts = bursts.where((str(t1.date()) < bursts)&(bursts < str(t2.date()))).dropna()

    df = pd.concat([pd.read_hdf('./data.hdf5',key=sat+"_"+datetime.strftime(pd.to_datetime(event[0]),format='%Y_%m_%dT%H_%M_%S')) for event in bursts.values]).dropna()
    ## Calculating velocity, and droping small density data

    df['ux']=(m_i*df['vx_i']+m_e*df['vx_e'])/(m_i+m_e)
    df['uy']=(m_i*df['vy_i']+m_e*df['vy_e'])/(m_i+m_e)
    df['uz']=(m_i*df['vz_i']+m_e*df['vz_e'])/(m_i+m_e)
    df = df.where(df['e_density']>1).dropna()
    if PINNS:
        df[['ex_mhd','ey_mhd','ez_mhd']] = E_mhd(df)
        df[['ex_hall','ey_hall','ez_hall']] = E_hall(df)
        ## Input / Output wanted
        input_features = ['bx', 'by', 'bz',
                    'jx', 'jy', 'jz',
                    'ux', 'uy', 'uz',
                    'ex_mhd','ey_mhd','ez_mhd',
                    'ex_hall','ey_hall','ez_hall',
                    'e_density',]
        output_targets = ['ex','ey','ez']
    else:
        ## Input / Output wanted
        input_features = ['bx', 'by', 'bz',
                    'jx', 'jy', 'jz',
                    'ux', 'uy', 'uz',
                    'e_density',]
        output_targets = ['ex','ey','ez']
    

=======
    # 4. Set the `torch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(mode=True)

    m_i = 1.67262192369e-27
    m_e = 9.1093837015e-31

    file = h5py.File(f'{data_path}','r')
    bursts = list(file[f"{sat}"].keys())
    file.close()
    bursts = pd.DataFrame([datetime.strptime(burst, f'%Y_%m_%dT%H_%M_%S') for burst in bursts])
    bursts = bursts.where((t1 < bursts)&(bursts < t2)).dropna()
    bursts = [datetime.strftime(event,'%Y_%m_%dT%H_%M_%S') for event in bursts[0]]

    df = pd.concat([pd.read_hdf(f'{data_path}',key=f"{sat}/{event}") for event in bursts]).dropna()
    ## Calculating velocity, and droping small density data
    df['ux']=(m_i*df['vx_i']+m_e*df['vx_e'])/(m_i+m_e)
    df['uy']=(m_i*df['vy_i']+m_e*df['vy_e'])/(m_i+m_e)
    df['uz']=(m_i*df['vz_i']+m_e*df['vz_e'])/(m_i+m_e)
    df = df.where(df['e_density']>density_threshold).dropna()
    if PINNS:
        df[['ex_mhd','ey_mhd','ez_mhd']] = E_mhd(df)
        df[['ex_hall','ey_hall','ez_hall']] = E_hall(df)
        input_features = ['bx', 'by', 'bz',
                'jx', 'jy', 'jz',
                'ux', 'uy', 'uz',
                'e_density',
                'ex_mhd','ey_mhd','ez_mhd',
                'ex_hall','ey_hall','ez_hall',
                ]
        output_targets = ['ex','ey','ez']
    else:
    ## Input / Output wanted
        input_features = ['bx', 'by', 'bz',
                    'jx', 'jy', 'jz',
                    'ux', 'uy', 'uz',
                    'e_density']
        output_targets = ['ex','ey','ez']
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
    df = df.drop(df.columns.drop(input_features+output_targets),axis=1) #drop useless data
    df = df[input_features + output_targets] #reorder columns for input to the left and output to the right

    ## Train interval 
    id_train_beg = 0
    id_train_end = int(len(df.index)*(f_train))

    t_train_begin = df.index.values[0]
    t_train_end = df.index.values[id_train_end]
    print("Train interval: ", t_train_begin, t_train_end)

    ## Validation interval

    id_val_beg = id_train_end+1
    id_val_end = int(len(df.index)*(f_train+f_valid))
    t_val_begin = df.index.values[id_val_beg]
    t_val_end = df.index.values[id_val_end]
    print("Validation interval: ", t_val_begin, t_val_end)

    ## Test interval 
    id_test_beg = id_val_end + 1
    id_test_end =  int(len(df.index))-1
    t_test_begin = df.index.values[id_test_beg]
    t_test_end = df.index.values[id_test_end]
    print("Test interval: ", t_test_begin, t_test_end)


<<<<<<< HEAD
    df_train, df_test = train_test_split(df,test_size = f_test+f_valid,train_size=f_train,random_state=seed,shuffle=False)
    df_val, df_test = df_test.iloc[:len(df_test)//2,:], df_test.iloc[len(df_test)//2:,:]

    # Divide the data before scaling
    df_train = df[id_train_beg:id_train_end]
    df_val = df[id_val_beg:id_val_end]
    df_test = df[id_test_beg:id_test_end]
=======
    df_train, df_test = train_test_split(df,test_size = f_test+f_valid,train_size=f_train,random_state=seed,shuffle=shuffle)
    df_val, df_test = df_test.iloc[:len(df_test)//2,:], df_test.iloc[len(df_test)//2:,:]

>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
    ## Define a scaler function
    scaler = MinMaxScaler()
    ## Obtain scaler based on the “train” data
    df_train_scaled = scaler.fit_transform(df_train)
    ## Apply the scaling obtained from the “train” data to “validation” and “test” data
    df_val_scaled = scaler.transform(df_val)
    df_test_scaled = scaler.transform(df_test)
<<<<<<< HEAD

=======
    
    #joblib.dump(scaler, f'{saved_models_path}/{name}/scaler.save') #Save the scale
>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
    ## Get input 'X'
    X_train = df_train_scaled[:,0:len(input_features)]
    X_test = df_test_scaled[:,0:len(input_features)]
    X_val = df_val_scaled[:,0:len(input_features)]
    ## Get output 'y'
    y_train = df_train_scaled[:,len(input_features):len(input_features)+len(output_targets)]
    y_test = df_test_scaled[:,len(input_features):len(input_features)+len(output_targets)]
    y_val = df_val_scaled[:,len(input_features):len(input_features)+len(output_targets)]

    model = first_dnn(input_features,output_targets)
    #model.summary()

<<<<<<< HEAD
    log_dir = f'../logs/fit/DNN_PINNS={PINNS}_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # checkpoint
    filepath= f'../Saved_models/DNN_PINNS={PINNS}/_weights.best.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    #X_train, y_train = lstm_data_transform(X_train,y_train)
    #X_val, y_val = lstm_data_transform(X_val,y_val)
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True)
    prof.start()
    model.fit(X_train, y_train, epochs=30, 
            validation_data=(X_val, y_val), verbose=2,
            callbacks=[tensorboard_callback, earlystopping_callback, checkpoint])

    prof.stop()
    model.save(f'../Saved_models/DNN_PINNS={PINNS}/model.h5')
=======
    log_dir = f'{log_path}/fit/{name}_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # checkpoint
    filepath= f'{saved_models_path}/{name}/_weights.best.keras'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    
    #X_train, y_train = lstm_data_transform(X_train,y_train)
    #X_val, y_val = lstm_data_transform(X_val,y_val)
    my_callbacks = [earlystopping_callback, checkpoint]
    model.fit(X_train, y_train, epochs=epochs, 
            validation_data=(X_val, y_val), verbose=2,
            callbacks=my_callbacks)
    model.save(f'{saved_models_path}/{name}/model.keras')

>>>>>>> 32b8d3e8c37ca9d32ec17387dc12328f779b0f32
