from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Input, Flatten
from tensorflow.python.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.activations import relu, softmax, sigmoid
from tensorflow.python.keras.metrics import categorical_accuracy

N_CLASS = 12


def palsol():
    i = Input(shape=(128, 32, 1))
    norm_i = BatchNormalization()(i)

    conv1 = Conv2D(filters=32, kernel_size=2, activation=relu)(norm_i)
    conv2 = Conv2D(filters=32, kernel_size=2, activation=relu)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = Dropout(rate=0.1)(pool1)

    conv3 = Conv2D(filters=48, kernel_size=3, activation=relu)(drop1)
    conv4 = Conv2D(filters=48, kernel_size=3, activation=relu)(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = Dropout(rate=0.2)(pool2)

    conv5 = Conv2D(filters=96, kernel_size=3, activation=relu)(drop2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv5)
    drop3 = Dropout(rate=0.2)(pool3)

    flat1 = Flatten()(drop3)
    dense1 = BatchNormalization()(Dense(units=128, activation=relu)(flat1))
    dense2 = BatchNormalization()(Dense(units=128, activation=relu)(dense1))
    out = Dense(units=N_CLASS, activation=softmax)(dense2)

    model = Model(inputs=[i], outputs=out)
    opt = optimizers.Adam()
    model.compile(
        optimizer=opt,
        # loss=losses.binary_crossentropy,
        loss=losses.categorical_crossentropy,
        metrics=[categorical_accuracy])
    return model


class SeResNet3:

    def __init__(self):
        i = Input(shape=(128, 32, 1))
        norm_i = BatchNormalization()(i)

        conv1 = Conv2D(filters=16, kernel_size=(3, 3))(norm_i)
        bn1 = BatchNormalization()(conv1)
        relu1 = relu(bn1)
        res1 = self.resblock(z=relu1, n_in=16, n_out=16)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(res1)
        drp1 = Dropout(rate=0.1)(pool1)

        conv2 = Conv2D(filters=32, kernel_size=(3, 3))(drp1)
        bn2 = BatchNormalization()(conv2)
        relu2 = relu(bn2)
        res2 = self.resblock(z=relu2, n_in=32, n_out=32)
        res3 = self.resblock(z=res2, n_in=32, n_out=32)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(res3)
        drp2 = Dropout(rate=0.2)(pool2)

        conv3 = Conv2D(filters=64, kernel_size=(3, 3))(drp2)
        bn3 = BatchNormalization()(conv3)
        relu3 = relu(bn3)
        res4 = self.resblock(z=relu3, n_in=64, n_out=64)
        res5 = self.resblock(z=res4, n_in=64, n_out=64)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(res5)
        drp3 = Dropout(rate=0.2)(pool3)

        conv4 = Conv2D(filters=128, kernel_size=(3, 3))(drp3)
        bn4 = BatchNormalization()(conv4)
        relu4 = relu(bn4)
        res6 = self.resblock(z=relu4, n_in=128, n_out=128)
        res7 = self.resblock(z=res6, n_in=128, n_out=128)
        drp4 = Dropout(rate=0.2)(res7)

        conv5 = Conv2D(filters=256, kernel_size=(3, 3))(drp4)
        bn5 = BatchNormalization()(conv5)
        relu5 = relu(bn5)
        pool5 = AveragePooling2D(pool_size=(2, 2))(relu5)

        flat1 = Flatten()(pool5)
        dense1 = Dense(units=256, activation=relu)(flat1)
        drp5 = Dropout(rate=0.2)(dense1)
        out = Dense(units=N_CLASS, activation=softmax)(drp5)

        model = Model(inputs=[i], outputs=out)
        opt = optimizers.Adam()
        model.compile(
            optimizer=opt,
            loss=losses.categorical_crossentropy,
            metrics=[categorical_accuracy])

        self.model = model

    def scale(self, z, n, red=16):
        pool = AveragePooling2D()(z)
        conv1 = Conv2D(red, (1, 1), activation=relu)(pool)
        conv2 = Conv2D(n, (1, 1))(conv1)
        return sigmoid(conv2)

    def resblock(self, z, n_in, n_out):
        conv1 = Conv2D(n_in, (3, 3))(z)
        bn1 = BatchNormalization()(conv1)
        relu1 = relu(bn1)
        conv2 = Conv2D(n_out, (3, 3))(relu1)
        bn2 = BatchNormalization(conv2)
        out = z + self.scale(bn2, n_out) * bn2
        return relu(out)
