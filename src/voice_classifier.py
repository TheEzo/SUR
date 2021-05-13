
from dataset import DatasetLoader

from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

import tensorflow as tf
import numpy as np
import kapre
import os
import re




class DataGenerator(tf.keras.utils.Sequence):
    def __init__( self, wav_paths, labels, sr, dt, n_classes, batch_size=32, shuffle=True ):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            X[i,] = path.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)



class VoiceClassifier:

    def __init__(self, dataset: DatasetLoader, epochs=80, batch_size=32):

        self.dataset = dataset
        
        self.patience = 20
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss = 'categorical_crossentropy'
        self.optimizer = SGD
        
        self.model_type = 'voice_classifier'
        self.dt = 1.0
        self.sr = 16000
        self.N_CLASSES = 31


    def build_model(self):
        self.classifier = self.conv_net_2d()
        print("Model built")


    def conv_net_2d(self):
        
        input_shape = (int( self.sr * self.dt), 1)

        input_layer = get_melspectrogram_layer( input_shape=input_shape, n_mels=128, pad_end=True, n_fft=512, win_length=400, hop_length=160, sample_rate= self.sr, 
                                                return_decibel=True, input_data_format='channels_last', output_data_format='channels_last')
        
        x_layer = LayerNormalization(axis = 2, name='batch_norm')(input_layer.output)
        
        x_layer = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x_layer)
        x_layer = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_0')(x_layer)
        
        x_layer = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_0')(x_layer)
        x_layer = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x_layer)

        #Was left out        
        x_layer = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_1')(x_layer)
        x_layer = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x_layer)
        
        x_layer = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x_layer)
        x_layer = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x_layer)
        #Was left out end

        x_layer = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x_layer)
        
        x_layer = layers.Flatten(name='flatten')(x_layer)
        x_layer = layers.Dropout(rate=0.2, name='dropout')(x_layer)
        x_layer = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x_layer)
        
        output_layer = layers.Dense(self.N_CLASSES, activation='softmax', name='softmax')(x_layer)
        
        model = Model(inputs = input_layer.input, outputs = output_layer, name = '2d_convolution')
        model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
        
        return model


    def train(self):

        model = self.classifier

        wav_train = self.dataset.x_train
        label_train = self.dataset.y_train - self.dataset.class_shift

        wav_val = self.dataset.x_val
        label_val = self.dataset.y_val - self.dataset.class_shift
        
        #COMMENT - if you want original train and val set distribution
            #Manually merged the train & val samples - so val above is basically empty
        wav_train, wav_val, label_train, label_val = train_test_split( wav_train, label_train, test_size=0.1, random_state=0)

        tg = DataGenerator(wav_train, label_train, self.sr, self.dt, self.N_CLASSES, self.batch_size )
        vg = DataGenerator(wav_val, label_val, self.sr, self.dt, self.N_CLASSES, self.batch_size )
    

        cp = ModelCheckpoint('snapshots/{}.h5'.format(self.model_type), monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
        earlystopping = EarlyStopping( patience = self.patience, monitor = "val_loss" )
        model.fit( tg, validation_data = vg, epochs=self.epochs, verbose = 1, callbacks = [cp, earlystopping] )



    def evaluate(self, voice_classifier_snapshot=None):

        if voice_classifier_snapshot is None:
            voice_classifier_snapshot = 'snapshots/{}.h5'.format(self.model_type)

        model = load_model( voice_classifier_snapshot, custom_objects={'STFT':STFT, 'Magnitude':Magnitude, 'ApplyFilterbank':ApplyFilterbank, 'MagnitudeToDecibel':MagnitudeToDecibel})

        wav_test = self.dataset.x_test
        label_test = self.dataset.x_names

        results = []
        batches = []
        occurences = self.dataset.inner_class_count( label_test )
        
        for x,y in zip(wav_test, label_test):
            
            y_pred = model.predict( x[np.newaxis] )
            y_pred = np.log( y_pred )
            
            #all batch-samples together
            if len(batches) < occurences[y]:
                batches.append( y_pred )

                if len(batches) == occurences[y]:
                    
                    #Depends how we want to interpret results..
                    y_mean = np.prod( np.array(batches), axis=0)   
                    #y_mean = np.mean(y_pred, axis=0)

                    y_pred_cls = np.argmax(y_mean) + self.dataset.class_shift
                    y_dist_cls = [ y, y_pred_cls ] + y_mean.tolist() 
                    
                    results.append( y_dist_cls )
                    batches = []
                    
                    print('ONE SAMPLE EVALUATED:')
                    print( y_dist_cls )

        return results
