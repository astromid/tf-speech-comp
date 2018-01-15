import os
import argparse
import models
from utils import TrainSequence2D, ValSequence2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.callbacks import TensorBoard
from utils import LoggerCallback
from keras_tqdm import TQDMCallback

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch', type=int)
parser.add_argument('--bal', type=int, default=1)
parser.add_argument('--eps', type=float, default=0.0)
parser.add_argument('--sil', type=float, default=0.1)
parser.add_argument('--unknown', type=float, default=0.1)
parser.add_argument('--aug', type=int, default=0)
parser.add_argument('--time', type=int, default=0)
parser.add_argument('--speed', type=float, default=0.0)
parser.add_argument('--vol', type=float, default=0.0)
parser.add_argument('--noise', type=float, default=0.0)
args = parser.parse_args()

ROOT_DIR = '..'
MODEL_DIR = os.path.join(ROOT_DIR, 'models', args.name)
LOGS_PATH = os.path.join(MODEL_DIR, 'logs')
EPOCHS = args.epochs
BATCH_SIZE = args.batch
TRAIN_PARAMS = {
    'batch_size': BATCH_SIZE,
    'balance': args.bal,
    'eps': args.eps,
    'silence': args.sil,
    'unknown': args.unknown,
    'augment': args.aug,
    'time_shift': args.time,
    'speed_tune': args.speed,
    'volume_tune': args.vol,
    'noise_vol': args.noise
}
os.makedirs(LOGS_PATH, exist_ok=True)

train_seq = TrainSequence2D(TRAIN_PARAMS)
val_seq = ValSequence2D(TRAIN_PARAMS)
# model = models.palsol()
model = models.SeResNet3().model

check_cb = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'model-best.h5'),
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
reduce_cb = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    factor=0.3,
    patience=5,
    verbose=1,
    epsilon=0.01,
    cooldown=3,
    min_lr=1e-6
)
tb_cb = TensorBoard(LOGS_PATH, batch_size=BATCH_SIZE)
log_cb = LoggerCallback()
tqdm_cb = TQDMCallback(leave_inner=False)
hist = model.fit_generator(
    generator=train_seq,
    steps_per_epoch=len(train_seq),
    epochs=EPOCHS,
    verbose=0,
    callbacks=[check_cb, reduce_cb, tb_cb, log_cb, tqdm_cb],
    validation_data=val_seq,
    validation_steps=len(val_seq)
)
model.save(os.path.join(MODEL_DIR, 'model.h5'))
print('Model saved successfully')
