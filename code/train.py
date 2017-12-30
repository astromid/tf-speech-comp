import numpy as np
import os
import argparse
import models
from utils import list_wav_files, train_generator, val_generator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size', default=64)
parser.add_argument('--silence', dest='max_silence_rate', default=0.2)
parser.add_argument('--epochs', dest='epochs', default=10)
parser.add_argument('--name', dest='name', default='palsol-1')
args = parser.parse_args()

ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
BATCH_SIZE = args.batch_size
MAX_SILENCE_RATE = args.max_silence_rate
LOGS_PATH = os.path.join(MODELS_DIR, args.name, 'logs')

os.makedirs(LOGS_PATH)

train_files, val_files = list_wav_files()
n_train = len(train_files)
n_val = len(val_files)
train_gen = train_generator(train_files, BATCH_SIZE, MAX_SILENCE_RATE)
val_gen = val_generator(val_files, BATCH_SIZE, MAX_SILENCE_RATE)
model = models.palsol_model()

# reduce_cb = ReduceLROnPlateau(monitor='val_acc')
# check_cb = ModelCheckpoint()
tb_cb = TensorBoard(LOGS_PATH, batch_size=BATCH_SIZE)

hist = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=np.ceil((n_train-6)/BATCH_SIZE).astype('int'),
    epochs=args.epochs,
    verbose=1,
    callbacks=[tb_cb],
    validation_data=val_gen,
    validation_steps=np.ceil(n_val/BATCH_SIZE).astype('int')
)

model.save(os.path.join(MODELS_DIR, args.name, 'model.h5'))
print('Model saved successfully')
