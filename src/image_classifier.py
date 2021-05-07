from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy, cosine_similarity
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import L1
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class ImageClassifier():
    def __init__(self, dataset, epochs=1000, batch_size=2):
        self.dataset = dataset
        
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss = cosine_similarity
        self.optimizer = Adam(lr=0.001)

    def build_model(self):
        self.classifier = self.build_classifier()
        #self.classifier.summary()
        self.classifier.compile(optimizer=self.optimizer, loss=self.loss, metrics=["categorical_accuracy"])

    def add_conv_block(self, inp, filters):
        cnn = Conv2D(filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_uniform")(inp)
        #cnn = BatchNormalization()(cnn)
        cnn = Conv2D(filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_uniform")(cnn)
        #cnn = BatchNormalization()(cnn)
        cnn = AveragePooling2D()(cnn)
        cnn = Dropout(0.3)(cnn)
        return cnn

    def add_dense_block(self, inp, neurons):
        cnn = Dense(neurons, activation="relu", kernel_initializer="he_uniform")(inp)
        #cnn = BatchNormalization()(cnn)
        cnn = Dense(neurons, activation="relu", kernel_initializer="he_uniform")(cnn)
        #cnn = BatchNormalization()(cnn)
        #cnn = Dropout(0.3)(cnn)
        return cnn

    def build_classifier(self):
        inp = Input(shape=self.dataset.image_shape) # 80x80

        cnn = self.add_conv_block(inp, 6) # 40x40
        cnn = self.add_conv_block(cnn, 6) # 20x20
        cnn = self.add_conv_block(cnn, 6) # 10x10
        #cnn = self.add_conv_block(cnn, 8) # 5x5

        cnn = Flatten()(cnn)
        out = Dense(self.dataset.class_count, activation="softmax", kernel_initializer="he_uniform")(cnn)

        return Model(inputs=inp, outputs=out)

    def load_weights(self, path):
        self.classifier.load_weights(path)

    def train(self, snapshot_path):
        # Create callbacks
        tensorboard_callback = TensorBoard(histogram_freq=1, write_images=True)
        modelcheckpoint_callback = ModelCheckpoint(snapshot_path, monitor="val_categorical_accuracy", save_best_only=True, save_weights_only=True)
        earlystopping_callback = EarlyStopping(patience=50, monitor="val_categorical_accuracy", baseline=0.3)

        # Change <1;31> labels to <0;30>
        y_val = np.subtract(self.dataset.y_val, self.dataset.class_shift)
        y_train = np.subtract(self.dataset.y_train, self.dataset.class_shift)

        # Data augmentation
        datagen = ImageDataGenerator(zoom_range=[0.9,1.1], brightness_range=[0.90,1.10])
        train_gen = datagen.flow(self.dataset.x_train, to_categorical(y_train), batch_size=self.batch_size)
        
        # Train model
        self.classifier.fit(self.dataset.x_train, to_categorical(y_train), batch_size=self.batch_size, epochs=self.epochs, callbacks=[earlystopping_callback, modelcheckpoint_callback], validation_data=(self.dataset.x_val, to_categorical(y_val)))
        
        # Load best weights and evaluate accuracy
        self.classifier.load_weights(snapshot_path)
        hist = self.classifier.evaluate(self.dataset.x_val, to_categorical(y_val), self.batch_size, return_dict=True)
        self.classifier.save_weights(f"_{hist['categorical_accuracy']:0.4f}.".join(snapshot_path.split(".")))
    
    def evaluate(self):
        evaluated_test = []
        for image, name in zip(self.dataset.x_test, self.dataset.x_names):
            pred = self.classifier.predict(image[np.newaxis])
            pred_idx = np.argmax(pred) + self.dataset.class_shift
            evaluated_test.append([name, pred_idx] + pred[0].tolist())
        
        return evaluated_test


