import os
import numpy as np
import pandas as pd
import librosa
import wavio
from PIL import Image
from librosa.core import to_mono



class DatasetLoader:
    def __init__(self, train_path, val_path, test_path, dataset_type):
        if dataset_type == "images":
            self.x_train, self.y_train = self.load_images_dataset(train_path)
            self.x_val, self.y_val = self.load_images_dataset(val_path)
            self.x_test, self.x_names = self.load_images_dataset(test_path, get_labels=False)
            self.check_image_dataset()
            self.class_shift = min(self.y_train) # We need to have labels starting from zero, not from one
        elif dataset_type == "voice":
            self.x_train, self.y_train = self.load_sound_dataset(train_path)
            self.x_val, self.y_val = self.load_sound_dataset(val_path)
            self.x_test, self.x_names = self.load_sound_dataset(test_path, get_labels=False)
            self.class_shift = min(self.y_train) # We need to have labels starting from zero, not from one
        else:
            print("Error: Unknown dataset type!")
            exit()

    def load_images_dataset(self, path, get_labels=True):
        images = []

        if get_labels:
            labels = []
            for image_class in os.scandir(path):
                for image in os.scandir(f"{path}/{image_class.name}/"):
                    if image.name.split(".")[-1] == "png": # Has .png ending
                        images.append(self.load_image(image.path))
                        labels.append(int(image_class.name)) # Class label
        else:
            file_names = []
            for image in os.scandir(f"{path}/"):
                if image.name.split(".")[-1] == "png": # Has .png ending
                    images.append(self.load_image(image.path))
                    file_names.append(".".join(image.name.split(".")[:-1]))

        if get_labels:
            return np.array(images), np.array(labels)
        else:
            return np.array(images), file_names


    def load_image(self, path):
        image = np.array(Image.open(path))
        image = (image - 127.5) / 127.5 # Convert to <-1;1> range
        return image


    def load_sound_dataset(self, path, get_labels=True):
        samples = []
        labels = []

        if get_labels:
            for item in os.scandir(path):
                for record in os.scandir(f'{path}/{item.name}/'):
                    if record.name.split('.')[-1] == 'wav':
                        
                        print(record.path)
                        print(item.name)
                        self.load_sound( record.path, int(item.name), samples, labels )
                        
        else:
            file_names = []
            for record in os.scandir(f'{path}/'):
                if record.name.split('.')[-1] == 'wav':
                    
                    item = ".".join(record.name.split(".")[:-1])
                    self.load_sound( record.path, item, samples, labels )

                    print(record.path)
                    print(item)
        
        print("Prepared dataset:")
        print(path)
        print( np.array(samples).shape )
        print( np.array(labels).shape )

        if get_labels:
            return np.array(samples), np.array(labels)
        else:
            return np.array(samples), labels



    def load_sound(self, record, item, samples, labels):

        threshold = 600
        delta_time = 1.0

        #Faster + all training done with this
        rate, wav = self.cut_noise(record, threshold)

        #Slow method, IDK why but does not learn sh*t
        # rate, wav = self.tezzo_cut_noise(record, sr)

        delta_sample = int( delta_time * rate )

        # cleaned audio is less than a single sample
        # pad with zeros to delta_sample size
        if wav.shape[0] < delta_sample:
            sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
            sample[:wav.shape[0]] = wav
            
            samples.append( sample )
            labels.append( item )

        # step through audio and save every delta_sample
        # discard the ending audio if it is too short
        else:
            trunc = wav.shape[0] % delta_sample
            for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                start = int(i)
                stop = int(i + delta_sample)
                sample = wav[start:stop]

                samples.append( sample )
                labels.append( item )



    def inner_class_count(self, data):
        unique, counts = np.unique(data, return_counts=True)
        return dict(zip(unique, counts))


    def check_image_dataset(self): # Check if dataset is correct
        assert self.x_train.shape[1:] == self.x_val.shape[1:] == self.x_test.shape[1:] # Same dimensions
        assert self.x_train.shape[0] == self.y_train.shape[0] # Same train sample count as labels
        assert self.x_val.shape[0] == self.y_val.shape[0] # Same val sample count as labels
        assert np.array_equal(np.unique(self.y_train), np.unique(self.y_val)) # Same classes in train and val

    @property
    def image_shape(self):
        return self.x_train.shape[1:] # Ignore sample count

    @property
    def class_count(self):
        return len(np.unique(self.y_train))

    @property
    def train_count(self):
        return self.x_train.shape[0]



    def cut_noise(self, path, threshold):

        sound = wavio.read(path)
        wav = sound.data.astype(np.float32, order='F')
        rate = sound.rate

        wav = to_mono(wav.reshape(-1))
        wav = wav.astype(np.int16)

        y = pd.Series(wav).apply(np.abs)
        y_mean = y.rolling( window = int(rate/20), min_periods = 1, center = True ).max()
        mask = [ True if mu > threshold else False for mu in y_mean ]

        wav = wav[mask]

        return rate, wav



    def tezzo_cut_noise(self, path, sr):
        margin_v = 10
        power = 2
        
        y, sr = librosa.load(path, mono=True, sr=sr)

        S_full, _ = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)

        mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power = power)
        
        # Once we have the masks, simply multiply them with the input spectrum to separate the components
        S_foreground = mask_v * S_full

        # apply mask
        yf = librosa.istft(S_foreground)

        yf2, _ = librosa.effects.trim(yf)

        return sr, yf2
