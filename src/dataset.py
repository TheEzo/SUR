import numpy as np
import os
from PIL import Image
from scipy.io import wavfile


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

    def load_sound_dataset(self, path, get_labels=True):
        sounds = []
        labels = []

        if get_labels:
            for item in os.scandir(f'{path}/'):
                for record in os.scandir(f'{path}/{item.name}/'):
                    if record.name.split('.')[-1] == 'wav':
                        sounds.append(self.load_sound(record.path))
                        labels.append(int(item.name))
            labels = np.array(labels)
        else:
            for record in os.scandir(f'{path}/'):
                if record.name.split('.')[-1] == 'wav':
                    sounds.append((self.load_sound(record.path)))
                    labels.append(".".join(record.name.split(".")[:-1]))

        return sounds, labels

    def load_image(self, path):
        image = np.array(Image.open(path))
        image = (image - 127.5) / 127.5 # Convert to <-1;1> range
        return image

    def load_sound(self, path):
        class Data:
            def __init__(self, samplerate, data):
                self.samplerate = samplerate
                self.data = data
        s, d = wavfile.read(path)
        return Data(s, d)

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


