from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

class ImageClassifier():
    def __init__(self, dataset, epochs=1000, batch_size=4):
        self.dataset = dataset
        
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss = categorical_crossentropy
        self.optimizer = Adam(lr=0.001)

    def build_model(self):
        self.classifier = self.build_classifier()
        self.classifier.summary()
        self.classifier.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

    def add_conv_block(self, inp, filters):
        cnn = Conv2D(filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_uniform")(inp)
        cnn = BatchNormalization()(cnn)
        cnn = Conv2D(filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_uniform")(inp)
        cnn = BatchNormalization()(cnn)
        cnn = AveragePooling2D()(cnn)
        cnn = Dropout(0.25)(cnn)
        return cnn

    def build_classifier(self):
        inp = Input(shape=self.dataset.image_shape) # 80x80

        cnn = self.add_conv_block(inp, 8) # 40x40
        cnn = self.add_conv_block(cnn, 16) # 20x20
        cnn = self.add_conv_block(cnn, 32) # 10x10
        cnn = self.add_conv_block(cnn, 64) # 5x5

        cnn = Flatten()(cnn)
        cnn = Dense(32, activation="relu", kernel_initializer="he_uniform")(cnn)
        out = Dense(self.dataset.class_count, activation="softmax", kernel_initializer="he_uniform")(cnn)

        return Model(inputs=inp, outputs=out)

    def train(self):
        tensorboard_callback = TensorBoard(histogram_freq=1, write_images=True)
        
        self.classifier.fit(self.dataset.x_train, to_categorical(self.dataset.y_train), batch_size=self.batch_size, epochs=self.epochs, callbacks=[tensorboard_callback], validation_data=(self.dataset.x_val, to_categorical(self.dataset.y_val)))
