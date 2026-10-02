from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import layers, regularizers, models



class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        model = models.Sequential()
        # Block 1
        model.add(layers.Conv2D(32, 3, activation="relu", input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        # Block 2
        model.add(layers.Conv2D(64, 3, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        # Block 3 (new)
        model.add(layers.Conv2D(128, 3, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        # Head
        model.add(layers.Flatten())
        model.add(layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(categories_count, activation="softmax"))
        self.model = model

    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )