import numpy as np
import os
import argparse
import models
from utils import list_wav_files, train_generator, val_generator
from utils import TrainSequence, ValSequence
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size', default=64)
parser.add_argument('--silence', dest='max_silence_rate', default=0.2)
parser.add_argument('--epochs', dest='epochs', default=10)
parser.add_argument('--name', dest='name', default='palsol-1')
parser.add_argument('--time', dest='time_shift', default=0)
parser.add_argument('--speed', dest='speed_tune', default=0)
parser.add_argument('--volume', dest='volume_tune', default=0)
parser.add_argument('--noise', dest='noise_vol', default=0)
parser.add_argument('--aug', dest='augment', default='yes')
args = parser.parse_args()

ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
LOGS_PATH = os.path.join(MODELS_DIR, args.name, 'logs')
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
TRAIN_PARAMS = {
    'batch_size': BATCH_SIZE,
    'augment': args.augment,
    'silence_rate': float(args.max_silence_rate),
    'time_shift': int(args.time_shift),
    'speed_tune': float(args.speed_tune),
    'volume_tune': float(args.volume_tune),
    'noise_vol': float(args.noise_vol)
}
os.makedirs(LOGS_PATH, exist_ok=True)

train_files, val_files, noise_files = list_wav_files()
n_train = len(train_files)
n_val = len(val_files)
#train_gen = train_generator(train_files, BATCH_SIZE, MAX_SILENCE_RATE)
#val_gen = val_generator(val_files, BATCH_SIZE, MAX_SILENCE_RATE)
train_seq = TrainSequence(train_files, noise_files, TRAIN_PARAMS)
val_seq = ValSequence(val_files, noise_files, TRAIN_PARAMS)
model = models.palsol_model()

# reduce_cb = ReduceLROnPlateau(monitor='val_acc')
# check_cb = ModelCheckpoint()
tb_cb = TensorBoard(LOGS_PATH, batch_size=BATCH_SIZE)

hist = model.fit_generator(
    generator=train_seq,
    # steps_per_epoch=np.ceil((n_train - 6) / BATCH_SIZE).astype('int'),
    steps_per_epoch=len(train_seq),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tb_cb],
    validation_data=val_seq,
    validation_steps=len(val_seq)
    # validation_steps=np.ceil(n_val / BATCH_SIZE).astype('int')
)

model.save(os.path.join(MODELS_DIR, args.name, 'model.h5'))
print('Model saved successfully')
