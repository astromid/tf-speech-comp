import os
import argparse
import models
from utils import TrainSequence2D, ValSequence2D
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')
parser.add_argument('--epochs', dest='epochs')
parser.add_argument('--batch', dest='batch_size')
parser.add_argument('--bal', dest='balance', default=0)
parser.add_argument('--eps', dest='eps', default=0)
parser.add_argument('--sil', dest='silence_rate', default=0)
parser.add_argument('--aug', dest='augment', default=0)
parser.add_argument('--time', dest='time_shift', default=0)
parser.add_argument('--speed', dest='speed_tune', default=0)
parser.add_argument('--vol', dest='volume_tune', default=0)
parser.add_argument('--noise', dest='noise_vol', default=0)
args = parser.parse_args()

ROOT_DIR = '..'
MODEL_DIR = os.path.join(ROOT_DIR, 'models', args.name)
LOGS_PATH = os.path.join(MODEL_DIR, 'logs')
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
TRAIN_PARAMS = {
    'batch_size': BATCH_SIZE,
    'balance': int(args.balance),
    'eps': float(args.eps),
    'silence_rate': float(args.silence_rate),
    'augment': int(args.augment),
    'time_shift': int(args.time_shift),
    'speed_tune': float(args.speed_tune),
    'volume_tune': float(args.volume_tune),
    'noise_vol': float(args.noise_vol),
}
os.makedirs(LOGS_PATH, exist_ok=True)

train_seq = TrainSequence2D(TRAIN_PARAMS)
val_seq = ValSequence2D(TRAIN_PARAMS)
# model = models.palsol()
model = models.SeResNet3().model

tb_cb = TensorBoard(LOGS_PATH, batch_size=BATCH_SIZE)
reduce_cb = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    patience=5,
    verbose=1
)
check_cb = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'model-best.h5'),
    monitor='val_categorical_accuracy',
    verbose=1,
    save_best_only=True
)

hist = model.fit_generator(
    generator=train_seq,
    steps_per_epoch=len(train_seq),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tb_cb, reduce_cb, check_cb],
    validation_data=val_seq,
    validation_steps=len(val_seq),
    max_queue_size=5,
    workers=1
)
model.save(os.path.join(MODEL_DIR, 'model.h5'))
print('Model saved successfully')
