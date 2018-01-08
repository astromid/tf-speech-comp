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
parser.add_argument('--balance', dest='balance', default=0)
parser.add_argument('--eps', dest='eps', default=0)
parser.add_argument('--silence', dest='silence_rate', default=0)
parser.add_argument('--aug', dest='augment', default=0)
parser.add_argument('--time', dest='time_shift', default=0)
parser.add_argument('--speed', dest='speed_tune', default=0)
parser.add_argument('--volume', dest='volume_tune', default=0)
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

# train_files, val_files, noise_files = list_wav_files()
# train_seq = TrainSequence(train_files, noise_files, TRAIN_PARAMS)
# val_seq = ValSequence(val_files, noise_files, TRAIN_PARAMS)
train_seq = TrainSequence2D(TRAIN_PARAMS)
val_seq = ValSequence2D(TRAIN_PARAMS)
model = models.palsol()

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
    verbose=1
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
    workers=2,
    use_multiprocessing=True
)
model.save(os.path.join(MODEL_DIR, 'model.h5'))
print('Model saved successfully')
