#!/usr/bin/env python
# coding: utf-8

# In[18]:



import vggish_main as vgg
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import math
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score
import itertools
# from python_speech_features import mfcc, logfbank
# Data augmentation
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[17]:


from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras_metrics as km

from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


# In[24]:


# For oversampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
import imblearn.under_sampling
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from collections import Counter


# ### Read Wave Files

# In[2]:


# Read wav files:
def read_wav(file_path):
    fs, data = wavfile.read(file_path)
    return data, fs


# ### Low Pass filter

# In[3]:


# Low pass filter using remez
def low_pass_remez(fs, cutoff, numtaps=400):
    trans_width = 100
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
    return taps


# ### Plotting the signal

# In[4]:


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)
    i=0
    for y in range(2):
        axes[0, y].set_title(list(signals.keys())[i])
        axes[0, y].plot(list(signals.keys())[i])
        axes[0, y].get_xaxis().set_visible(False)
        axes[0, y].get_yaxis().set_visible(False)
        i+=1
            


# In[5]:


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16)
    i=0
    for y in range(2):
        data=list(fft.values())[i]
        Y, freq=data[0], data[1]
        axes[0, y].set_title(list(fft.keys())[i])
        axes[0, y].plot(freq, Y)
        axes[0, y].get_xaxis().set_visible(False)
        axes[0, y].get_yaxis().set_visible(False)
        i+=1


# In[6]:


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i=0
    for x in range(1):
        for y in range(2):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i+=1
            


# In[7]:


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('MFCC', size=16)
    i=0
    for x in range(1):
        for y in range(2):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i+=1
            


# ### Audio features (fft, mfcc...)

# In[8]:


def calc_fft(y, rate):
    n=len(y)
    freq=np.fft.rfftfreq(n, d=1/rate)
    Y=abs(np.fft.rfft(y)/n)
    return (Y, freq)


# In[10]:


def envelope(y, rate, threshold):
    mask=[]
    y= pd.Series(y).apply(np.abs)
    y_mean=y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
    


# # Data Augmentation

# In[ ]:


# AddGaussianNoise, TimeStretch, PitchShift, Shift


# In[27]:


# Add Gaussian Noise
def add_gaussian_noise(data_path, file_info, n_repeats=3, min_amp=0.001, max_amp=0.015):
    # Create the augmenter
    augmenter = Compose([AddGaussianNoise(min_amplitude=min_amp, max_amplitude=max_amp, p=1.0)])
    
    # Iterate through the Gibbon audio files only
    for j in file_info[file_info.label==1].index:
        for i in range(n_repeats):
            # Read audio file
            rate, samples = wavfile.read(data_path + 'Clean/' + file_info.at[j, 'fname'])
            # Set the output path
            output_file_path = data_path + 'Augmented/AddGaussianNoise_{:03d}_'.format(i) + file_info.at[j, 'fname']
            # Add gaussian noise
            augmented_samples = augmenter(samples=samples, sample_rate=rate)
            # Save the new audio
            wavfile.write(filename=output_file_path, rate = rate, data = augmented_samples)


# In[28]:


# Time Stretch
def time_stretch(data_path, file_info, n_repeats=3, min_rate=0.8, max_rate=1.25):
    # Create the augmenter
    augmenter = Compose([TimeStretch(min_rate = min_rate, max_rate=max_rate, p=1.0)])
    
    # Iterate through the Gibbon audio files only
    for j in file_info[file_info.label==1].index:
        for i in range(n_repeats):
            # Read audio file
            rate, samples = wavfile.read(data_path + 'Clean/' + file_info.at[j, 'fname'])
            # Set the output path
            output_file_path = data_path + 'Augmented/TimeStretch_{:03d}_'.format(i) + file_info.at[j, 'fname']
            # Perform time stretch
            augmented_samples = augmenter(samples=samples, sample_rate=rate)
            # Save the new audio
            wavfile.write(filename=output_file_path, rate = rate, data = augmented_samples)


# In[29]:


# Pitch Shift
def pitch_shift(data_path, file_info, n_repeats=3, min_semitones=-4, max_semitones=4):
    # Create the augmenter
    augmenter = Compose([PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=1.0)])
    
    # Iterate through the Gibbon audio files only
    for j in file_info[file_info.label==1].index:
        for i in range(n_repeats):
            # Read audio file
            rate, samples = wavfile.read(data_path + 'Clean/' + file_info.at[j, 'fname'])
            # Set the output path
            output_file_path = data_path + 'Augmented/PitchShift_{:03d}_'.format(i) + file_info.at[j, 'fname']
            # Perform time stretch
            augmented_samples = augmenter(samples=samples, sample_rate=rate)
            # Save the new audio
            wavfile.write(filename=output_file_path, rate = rate, data = augmented_samples)


# In[30]:


# Shift
def shift(data_path, file_info, n_repeats=3, min_fraction=-0.5, max_fraction=0.5):
    # Create the augmenter
    augmenter = Compose([Shift(min_fraction = min_fraction, max_fraction = max_fraction, p=1.0)])
    
    # Iterate through the Gibbon audio files only
    for j in file_info[file_info.label==1].index:
        for i in range(n_repeats):
            # Read audio file
            rate, samples = wavfile.read(data_path + 'Clean/' + file_info.at[j, 'fname'])
            # Set the output path
            output_file_path = data_path + 'Augmented/Shift_{:03d}_'.format(i) + file_info.at[j, 'fname']
            # Perform time stretch
            augmented_samples = augmenter(samples=samples, sample_rate=rate)
            # Save the new audio
            wavfile.write(filename=output_file_path, rate = rate, data = augmented_samples)


# ### Create NN models

# In[ ]:


def get_binary_model(input_shape=128):
    model = Sequential()
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal', input_dim=input_shape))
    model.add(Dense(256, activation='relu', kernel_initializer='random_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['acc', km.binary_precision(), km.binary_recall()])
    
    return model
    


# In[ ]:


def get_conv_model(input_shape):
    model=Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['acc', km.binary_precision(), km.binary_recall()])
    
    return model


# In[ ]:


def get_recurrent_model(input_shape):
    model=Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['acc', km.binary_precision(), km.binary_recall()])
    
    return model


# In[ ]:


def get_CNN_recurrent_model(input_shape):
    model=Sequential()
    # CNN model
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                                     padding='same', input_shape=input_shape)))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same')))
    model.add(TimeDistributed(MaxPool2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    
    # LSTM model
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[14]:


# Another method for CNN-LSTM
# def method2(input_shape):
#     cnn = Sequential()
#     cnn.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
#                                      padding='same', input_shape=input_shape))
#     cnn.add(MaxPool2D((2, 2)))
#     cnn.add(Flatten())
#     # define LSTM model
#     model = Sequential()
#     model.add(TimeDistributed(cnn, input_shape=(None, 2, 224, 224, 13)))
#     model.add(LSTM(128, return_sequences=True))
#     model.add(Dense(1, activation='sigmoid'))
    
#     model.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model
# #     You aso need to specify a batch size in the input dimensions to that layer I guess,
# #     to get the fifth dimension. Try using:
# #     model.add(TimeDistributed(cnn, input_shape=(None, num_timesteps, 224, 224,num_chan))).
# #     The None will then allow variable batch size.


# ### Predictions

# In[ ]:


def build_predictions(directory):
    y_true=[]
    y_pred=[]
    
    classifier = load_model('vggish_binary_class_model.h5')
    y_pred_unseen = classifier.predict(postprocessed_batch_appended)
    y_pred_unseen = (y_pred_unseen>0.5)
    # Accuracy of the unseen data
    cm_unseen = confusion_matrix(y[:, 0], y_pred_unseen)
    print(cm_unseen)


# ### Extract VGG features

# In[ ]:


def extract_vgg_features(file_info, data_path, hop_size=0.96):
    # Iterate through the files
    for i in tqdm(file_info.index):
        # If the file has already been converted skip it
        if(file_info.at[i, 'vgg_done']==1):
            continue
        try:
            vgg_graph = tf.Graph()
            with vgg_graph.as_default():
                session_vgg = tf.Session()
                with session_vgg.as_default():
                    vgg_net = vgg.CreateVGGishNetwork(session_vgg, hop_size = hop_size)
                    vgg_mfcc, _ = vgg.ProcessWithVGGish(session_vgg, vgg_net,
                                                        data_path+'Clean/'+file_info.at[i, 'fname'])
                    # Save the vgg features locally on machine
                    with open(data_path+'VGG/' + file_info.at[i, 'fname'], 'wb') as f:
                        pickle.dump(vgg_mfcc, f)
                    file_info.at[i, 'vgg_done']=1
            # Save the updated file info and return it
            file_info.loc[:, file_info.columns!='length'].to_csv(data_path+'files_labels.csv', index=False)
        except:
            print('Failure: ', i)
    return file_info

def extract_vgg_features_1(file_info, data_path, hop_size=0.96):
    # Iterate through the files
    for i in tqdm(file_info.index):
        # If the file has already been converted skip it
        if(file_info.at[i, 'vgg_done']==1):
            continue
        try:
            vgg_graph = tf.Graph()
            with vgg_graph.as_default():
                session_vgg = tf.Session()
                with session_vgg.as_default():
                    vgg_net = vgg.CreateVGGishNetwork(session_vgg, hop_size = hop_size)
                    vgg_mfcc, _ = vgg.ProcessWithVGGish(session_vgg, vgg_net,
                                                        data_path + '/' + file_info.at[i, 'fpath'])
                    # Save the vgg features locally on machine
                    with open(data_path+'VGG_hop_10/' + file_info.at[i, 'fname'], 'wb') as f:
                        pickle.dump(vgg_mfcc, f)
                    file_info.at[i, 'vgg_done']=1
            # Save the updated file info and return it
            file_info.loc[:, file_info.columns!='length'].to_csv(data_path+'files_labels_unmasked_hop_10.csv', index=False)
        except:
            print('Failure: ', i)
    return file_info

def extract_vgg_features_folder(path):
    # Iterate through the files
    for file in os.listdir(path):
        if file.endswith(".wav"):
            try:
                vgg_graph = tf.Graph()
                with vgg_graph.as_default():
                    session_vgg = tf.Session()
                    with session_vgg.as_default():
                        vgg_net = vgg.CreateVGGishNetwork(session_vgg)
                        vgg_mfcc, _ = vgg.ProcessWithVGGish(session_vgg, vgg_net, path + file)
                        # Save the vgg features locally on machine
                        with open(path + 'VGG/' + file, 'wb') as f:
                            pickle.dump(vgg_mfcc, f)
            except:
                print('Failure: ', i)
                
# # Resampling Methods

# In[25]:


# Random Undersampling
def random_undersampling(data_X, data_y, ratio):
    # The ratio represents the amount of data in the minority class compared to the majority one
    rand_under_samp = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=ratio)
    under_samp_X, under_samp_y = rand_under_samp.fit_resample(data_X, data_y)
    
    return under_samp_X, under_samp_y


# In[26]:


# SMOTE and EN resmapling
def smote_en_resampling(data_X, data_y, k_neighbors=5):
    # Perform under and over sampling using SMOTE and EN
    smote = SMOTE(sampling_strategy='minority', k_neighbors=k_neighbors, n_jobs=8)
    enn = EditedNearestNeighbours(n_neighbors=k_neighbors, n_jobs=8)
    smoteen = SMOTEENN(sampling_strategy="minority", smote=smote, enn=enn, n_jobs=8)
    resamp_X, resamp_y = smoteen.fit_sample(data_X, data_y)
    
    return resamp_X, resamp_y


# # Cross Validation

# In[20]:


# Randomly select files for test and train data
# This function returns the file names only
def build_rand_feat(file_info, training_ratio=0.75):
    training_files=[]
    unseen_files=[]
    
    rand_files=file_info.groupby(['label'])['fname'].apply(lambda x: np.random.permutation(x))

    for j in range(rand_files.shape[0]):
        # Get the number of files in the training and testing variables
        files_cnt=len(rand_files[j])
        train_samp=math.ceil(files_cnt*training_ratio)

        training_files.append(rand_files[j][0:train_samp])
        unseen_files.append(rand_files[j][train_samp:files_cnt])
    
    # Convert them to a 1d list
    training_files=list(itertools.chain.from_iterable(training_files))
    unseen_files=list(itertools.chain.from_iterable(unseen_files))
    
    # Shuffle the data
    training_files=np.random.permutation(training_files)
    unseen_files=np.random.permutation(unseen_files)
    
    return training_files, unseen_files


# In[23]:


# Overlap the data for LSTM
# https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

# Perform cross validation
# model_type can be: lstm, binary, knn
def cross_validation(model_type, file_info, training_files, data_path, model_name_prefix,
                     nfolds=5, lstm_steps=[], thresholds=[0.5], apply_class_weight=True,
                     rand_under_samp=True, under_samp_ratio=0.3, smote_en_resamp=True, smote_en_neighbors=5,
                     include_augmented=False, k_neighbors=5):
    # Get the number of testing files based on the number of folds
    num_files = len(training_files)
    num_test_files = math.floor(num_files/nfolds)
    test_start_index = 0
    model = None
    
    conf_matrix = []
    recall = []
    precision = []
    fscore = []
    
    for i in range(nfolds):
        print('####### Fold: ', i)
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        
        # set the starting index
        test_start_index = i * num_test_files
        # Set the testing and training indexes
        test_ind = range(test_start_index, (test_start_index + num_test_files - 2))
        # Get the indexes that are not in the testing range
        train_ind = [ind for ind in range(num_files) if ind not in test_ind]
        
        print('####### Reading testing data')
        # Read the testing data
        for j in test_ind:
            # Get the file name and read the data
            file = file_info[file_info.fname == training_files[j]].fname.iloc[0]
            audio_data = pickle.load(open(data_path + 'VGG/'+ file, 'rb'))
            test_X.append(audio_data)
            # Get the class label
            label = file_info[file_info.fname == training_files[j]].label.iloc[0]
            test_y.append(np.repeat([label], (len(audio_data))))
        
        print('####### Reading training data')
        # Read the training data
        for j in train_ind:
            # Get the file name and read the data
            file = file_info[file_info.fname == training_files[j]].fname.iloc[0]
            audio_data = pickle.load(open(data_path + 'VGG/' + file, 'rb'))
            train_X.append(audio_data)
            # Get the class label
            label = file_info[file_info.fname == training_files[j]].label.iloc[0]
            train_y.append(np.repeat([label], len(audio_data)))
        
        # Read the Augmented data
        # All the augmented data are of class 1 (Gibbon calls)
        if include_augmented == True:
            # Get the file name and read the data
            for file in os.listdir(data_path + 'Augmented/VGG'):
                if file.endswith(".wav"):
                    audio_data = pickle.load(open(data_path + 'Augmented/VGG/' + file, 'rb'))
                    train_X.append(audio_data)
                    train_y.append(np.repeat(1, len(audio_data)))
        
        # Reshape the arrays to 1D
        train_X = [response for sublist in train_X for response in sublist]
        train_y = [response for sublist in train_y for response in sublist]
        test_X = [response for sublist in test_X for response in sublist]
        test_y = [response for sublist in test_y for response in sublist]

        print('######## Before resampling, num Positives (train): ', sum(train_y), '/', len(train_y))
        print('######## Before resampling, num Positives (test): ', sum(test_y), '/', len(test_y))
        
        # Perform resampling techniques inside the cross validation to maximise randomness
        # and reduce over-fitting
        if rand_under_samp == True:
            print('####### Random Undersamp')
            train_X, train_y = random_undersampling(train_X, train_y, under_samp_ratio)
            print('######## After random undersampling, num Positives (train): ', sum(train_y), '/', len(train_y))
            print('######## After random undersampling, num Positives (test): ', sum(test_y), '/', len(test_y))
        
        if smote_en_resamp == True:
            print('####### Smote_En')
            train_X, train_y = smote_en_resampling(train_X, train_y, smote_en_neighbors)
            print('######## After SMOTE, num Positives (train): ', sum(train_y), '/', len(train_y))
            print('######## After SMOTE, num Positives (test): ', sum(test_y), '/', len(test_y))
        
        # Save the data
        with open(data_path + 'CV_Data/' + model_name_prefix + '_train_CV' + str(i), 'wb') as f:
            pickle.dump([train_X, train_y], f)
        with open(data_path + 'CV_Data/' + model_name_prefix + '_test_CV' + str(i), 'wb') as f:
            pickle.dump([test_X, test_y], f)
        print('####### Data saved')
        
        # Fit the model
        print('####### Model fitting')
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)

        if model_type == 'lstm':
            for step in lstm_steps:
                print('####### Step: ', step)
                # Reshape the input data
                train_X_step, train_y_step = temporalize(X=train_X, y=train_y, lookback=step)
                train_X_step = np.array(train_X_step)
                train_y_step = np.array(train_y_step)
                
#                 train_X_step = np.array(train_X)
#                 train_y_step = np.array(train_y)
                # Get a valid length to reshape
#                 print('#################################################################')
#                 print('################# ', train_X_step.shape, '   ###################')
#                 print('################# ', train_y_step.shape, '   ###################')
#                 print('#################################################################')
            
#                 size_train = math.floor(train_X_step.shape[0] / step)
#                 train_X_step = train_X_step[:size_train]
#                 train_y_step = train_y_step[:size_train]
            
                train_X_step = train_X_step.reshape(-1, step, 128)
                input_shape = (step, 128)
                
                test_X_step = np.array(test_X)
                test_y_step = np.array(test_y)
                test_X_step, test_y_step = temporalize(X=test_X_step, y=test_y_step, lookback=step)
                test_X_step = np.array(test_X_step)
                test_y_step = np.array(test_y_step)
                
#                 # Get a valid length to reshape
#                 size_test = math.floor(test_X_step.shape[0] / step)
#                 test_X_step = test_X_step[:size_test]
#                 test_y_step = test_y_step[:size_test]
                
#                 print('#################################################################')
#                 print('################# ', size_train, '   ###################')
#                 print('################# ', train_X_step.shape, '   ###################')
#                 print('################# ', train_y_step.shape, '   ###################')
#                 print('################# ', list(np.unique(train_y_step)), '   ###################')
#                 print('#################################################################')
#                 print('################# ', size_test, '   ###################')
#                 print('################# ', test_X_step.shape, '   ###################')
#                 print('################# ', test_y_step.shape, '   ###################')
#                 print('################# ', list(np.unique(test_y_step)), '   ###################')
#                 print('#################################################################')
#                 print(train_X_step)
#                 print('#################################################################')
#                 print(train_y_step)
#                 print('#################################################################')
                
                test_X_step = test_X_step.reshape(-1, step, 128)
                
                # Get the model and train it
                model = get_recurrent_model(input_shape)
                if apply_class_weight == True:
                    classes = list(np.unique(file_info.label))
                    class_weight = compute_class_weight('balanced', classes, train_y_step)
                    model.fit(train_X_step, train_y_step, epochs=50, batch_size=32,
                              class_weight=class_weight, verbose=0)
                else:
                    model.fit(train_X_step, train_y_step, epochs=50, batch_size=32, verbose=0)
                # Save the model
                class_file_name = model_name_prefix + '_CV' + str(i) + '_step' + str(step) + '.h5'
                model.save(data_path + 'Models/' + class_file_name)
                print('####### Model saved')
                
                print('####### Model testing')
                # Testing the model
                y_pred = model.predict(test_X_step)
                for threshold in thresholds:
                    print('####### Threshold: ', threshold)
                    y_pred_th = (y_pred > threshold)
                    
                    # Confusion matrix
                    cm_test = confusion_matrix(test_y_step, y_pred_th)
                    conf_matrix = np.append(conf_matrix, cm_test)
                    print(cm_test)

                    # Recall
                    tpr_test = recall_score(test_y_step, y_pred_th)
                    recall = np.append(recall, tpr_test)
                    print('TPR: ', tpr_test)

                    # Precision
                    precision_test = precision_score(test_y_step, y_pred_th)
                    precision = np.append(precision, precision_test)
                    print('Precision: ', precision_test)

                    # F1 score
                    fscore_test = f1_score(test_y_step, y_pred_th)
                    fscore = np.append(fscore, fscore_test)
                    print('F1 score: ', fscore_test)

        else:
            if model_type == 'binary':
                # Get the model and train it
                model = get_binary_model()
                if apply_class_weight == True:
                    classes = list(np.unique(file_info.label))
                    class_weight = compute_class_weight('balanced', classes, train_y)
                    history = model.fit(train_X, train_y, epochs=50, batch_size=32,
                              class_weight=class_weight)
                    plt.plot(history.history['acc'])
                    plt.plot(history.history['val_acc'])
                    plt.title('model accuracy')
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.show()
                else:
                    model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=0)
                # Save the model
                class_file_name = model_name_prefix + '_CV' + str(i) + '.h5'
                model.save(data_path + 'Models/' + class_file_name)

            if model_type == 'knn':
                # Rescale the x values for train and test
                scaler = StandardScaler()
                scaler.fit(train_X)
                train_X = scaler.transform(train_X)
                test_X = scaler.transform(test_X)
                # Get the model and train it
                model = KNeighborsClassifier(n_neighbors=k_neighbors)
                model.fit(train_X, train_y)
            # Save the model
            class_file_name = model_name_prefix + '_CV' + str(i) + '.h5'
            with open(data_path + 'Models/' + class_file_name, 'wb') as f:
                    pickle.dump(model, f)
            
            print('####### Model saved')

            print('####### Model testing')
            # Testing the model
            y_pred = model.predict(test_X)
            for threshold in thresholds:
                print('####### Threshold: ', threshold)
                y_pred_th = (y_pred > threshold)

                # Confusion matrix
                cm_test = confusion_matrix(test_y, y_pred_th)
                conf_matrix = np.append(conf_matrix, cm_test)
                print(cm_test)

                # Recall
                tpr_test = recall_score(test_y, y_pred_th)
                recall = np.append(recall, tpr_test)
                print('TPR: ', tpr_test)

                # Precision
                precision_test = precision_score(test_y, y_pred_th)
                precision = np.append(precision, precision_test)
                print('Precision: ', precision_test)

                # F1 score
                fscore_test = f1_score(test_y, y_pred_th)
                fscore = np.append(fscore, fscore_test)
                print('F1 score: ', fscore_test)

    return conf_matrix, recall, precision, fscore


# In[ ]:



def predic_unseen(model_name, file_info, unseen_files, data_path, threshold=0.5):
    # Load Model
    dependencies = {
        'binary_precision': km.binary_precision(),
        'binary_recall': km.binary_recall()

    }
    model = load_model(data_path + 'Models/' + model_name, custom_objects=dependencies)

    # Load data
    test_x = []
    test_y = []
    for j in range(len(unseen_files)):
        # Get the file name and read the data
        file = file_info[file_info.fname == unseen_files[j]].fname.iloc[0]
        audio_data = pickle.load(open(data_path + 'VGG/' + file, 'rb'))
        test_x.append(audio_data)
        # Get the class label
        label = file_info[file_info.fname == unseen_files[j]].label.iloc[0]
        test_y.append(np.repeat([label], (len(audio_data))))

    # Make predictions
    test_x = [response for sublist in test_x for response in sublist]
    test_y = [response for sublist in test_y for response in sublist]
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    y_pred = model.predict(test_x)
    y_pred_th = (y_pred > threshold)

    # Confusion matrix
    cm_test = confusion_matrix(test_y, y_pred_th)
    print(cm_test)

    # Recall
    tpr_test = recall_score(test_y, y_pred_th)
    print('TPR: ', tpr_test)

    # Precision
    precision_test = precision_score(test_y, y_pred_th)
    print('Precision: ', precision_test)

    # F1 score
    fscore_test = f1_score(test_y, y_pred_th)
    print('F1 score: ', fscore_test)


# Feature analysis:

# Get the mean difference between the features values of each class
def get_distance_means(files):
    gibbon_mean_features = []
    noise_mean_features = []
    
    # Iterate through each file and get the average of the features for each class
    for audio_num in range(1, 7):
        gibbon_sum = 0
        noise_sum = 0
        gibbon_cnt = 0
        noise_cnt = 0
        for file in files.index:
            if files.loc[file].fname.startswith(str(audio_num)):
                features = pickle.load(open(data_path + 'VGG/' + files.loc[file].fname, 'rb'))
                if files.loc[file].label == 1:
                    # If gibbon features
                    gibbon_sum = gibbon_sum + np.mean(features)
                    gibbon_cnt = gibbon_cnt + 1
                if files.loc[file].label == 0:
                    # If noise features
                    noise_sum = noise_sum + np.mean(features)
                    noise_cnt = noise_cnt + 1

        gibbon_mean_features.append(gibbon_sum / gibbon_cnt)
        noise_mean_features.append(noise_sum / noise_cnt)
    
    return gibbon_mean_features, noise_mean_features
    
    
# Calculate the difference between each value of 2 vectors
def create_difference_matrix(a, b):
    difference = pd.DataFrame()
    for i in range(len(a)):
        diff = []
        for j in range(len(b)):
            diff.append(abs(a[i] - b[j]))
        difference[str(i + 1)] = diff
    
    return difference
    

