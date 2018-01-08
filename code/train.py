import os
import argparse
import models
# from utils import list_wav_files, TrainSequence, ValSequence
from utils import TrainSequence2D, ValSequence2D
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size')
parser.add_argument('--silence', dest='max_silence_rate', default=0.2)
parser.add_argument('--epochs', dest='epochs')
parser.add_argument('--name', dest='name')
parser.add_argument('--time', dest='time_shift', default=0)
parser.add_argument('--speed', dest='speed_tune', default=0)
parser.add_argument('--volume', dest='volume_tune', default=0)
parser.add_argument('--noise', dest='noise_vol', default=0)
parser.add_argument('--aug', dest='augment', default='no')
parser.add_argument('--balance', dest='balance', default='no')
parser.add_argument('--eps', dest='eps', default=0)
args = parser.parse_args()

ROOT_DIR = '..'
MODEL_DIR = os.path.join(ROOT_DIR, 'models', args.name)
LOGS_PATH = os.path.join(MODEL_DIR, 'logs')
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
TRAIN_PARAMS = {
    'batch_size': BATCH_SIZE,
    'augment': args.augment,
    'silence_rate': float(args.max_silence_rate),
    'time_shift': int(args.time_shift),
    'speed_tune': float(args.speed_tune),
    'volume_tune': float(args.volume_tune),
    'noise_vol': float(args.noise_vol),
    'balance': args.balance,
    'eps': float(args.eps)
}
os.makedirs(LOGS_PATH, exist_ok=True)

# train_files, val_files, noise_files = list_wav_files()
# n_train = len(train_files)
# n_val = len(val_files)
# train_seq = TrainSequence(train_files, noise_files, TRAIN_PARAMS)
# val_seq = ValSequence(val_files, noise_files, TRAIN_PARAMS)
train_seq = TrainSequence2D(TRAIN_PARAMS)
val_seq = ValSequence2D(TRAIN_PARAMS)
model = models.palsol_model()

check_cb = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'model-best.h5'),
    monitor='val_categorical_accuracy',
    verbose=1,
    save_best_only=True
)
tb_cb = TensorBoard(LOGS_PATH, batch_size=BATCH_SIZE)
reduce_cb = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    patience=5,
    verbose=1,
    min_lr=1e-6
)

hist = model.fit_generator(
    generator=train_seq,
    steps_per_epoch=len(train_seq),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[check_cb, tb_cb, reduce_cb],
    validation_data=val_seq,
    validation_steps=len(val_seq),
    max_queue_size=20,
    workers=2
)
model.save(os.path.join(MODEL_DIR, 'model.h5'))
print('Model saved successfully')
