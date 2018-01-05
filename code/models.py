from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Input, Flatten
from tensorflow.python.keras.layers import Dropout, BatchNormalization, MaxPooling2D
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.metrics import categorical_accuracy

N_CLASS = 12


def palsol_model():
    i = Input(shape=(128, 32, 1))
    norm_i = BatchNormalization()(i)

    conv1 = Conv2D(32, kernel_size=2, activation=relu)(norm_i)
    conv2 = Conv2D(32, kernel_size=2, activation=relu)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = Dropout(rate=0.1)(pool1)

    conv3 = Conv2D(48, kernel_size=3, activation=relu)(drop1)
    conv4 = Conv2D(48, kernel_size=3, activation=relu)(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = Dropout(rate=0.2)(pool2)

    conv5 = Conv2D(96, kernel_size=3, activation=relu)(drop2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv5)
    drop3 = Dropout(rate=0.2)(pool3)

    flat1 = Flatten()(drop3)
    dense1 = BatchNormalization()(Dense(128, activation=relu)(flat1))
    dense2 = BatchNormalization()(Dense(128, activation=relu)(dense1))
    out = Dense(N_CLASS, activation=softmax)(dense2)

    model = Model(inputs=[i], outputs=out)
    opt = optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=losses.binary_crossentropy,
        metrics=['acc', categorical_accuracy])
    return model
