import os
import shutil

original_dataset_dir0 = '/home/bater/PycharmProjects/нейро/keras-tutorial/prep/car'
original_dataset_dir1 = '/home/bater/PycharmProjects/нейро/keras-tutorial/prep/wall'
original_dataset_dir2 = '/home/bater/PycharmProjects/нейро/keras-tutorial/prep/porebrik'

base_dir = '/home/bater/PycharmProjects/нейро/keras-tutorial/prep/save'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_car_dir = os.path.join(train_dir, 'car')
os.mkdir(train_car_dir)
train_wall_dir = os.path.join(train_dir, 'wall')
os.mkdir(train_wall_dir)
train_porebrik_dir = os.path.join(train_dir, 'porebrik')
os.mkdir(train_porebrik_dir)

validation_car_dir = os.path.join(validation_dir, 'car')
os.mkdir(validation_car_dir)
validation_wall_dir = os.path.join(validation_dir, 'wall')
os.mkdir(validation_wall_dir)
validation_porebrik_dir = os.path.join(validation_dir, 'porebrik')
os.mkdir(validation_porebrik_dir)

train_car_dir = os.path.join(train_dir, 'car')
os.mkdir(train_car_dir)
train_wall_dir = os.path.join(train_dir, 'wall')
os.mkdir(train_wall_dir)
train_porebrik_dir = os.path.join(train_dir, 'porebrik')
os.mkdir(train_porebrik_dir)

fnames = ['{}.jpg'.format(i) for i in range(200)]
for fname in fnames:
    src = os.path.join(original_dataset_dir0, fname)
    dst = os.path.join(train_car_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(200, 400)]
for fname in fnames:
    src = os.path.join(original_dataset_dir0, fname)
    dst = os.path.join(validation_car_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(400, 579)]
for fname in fnames:
    src = os.path.join(original_dataset_dir0, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(200)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(train_wall_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(200, 300)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(validation_wall_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(300, 381)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(test_wall_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(200)]
for fname in fnames:
    src = os.path.join(original_dataset_dir2, fname)
    dst = os.path.join(train_porebrik_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(200, 300)]
for fname in fnames:
    src = os.path.join(original_dataset_dir2, fname)
    dst = os.path.join(validation_porebrik_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(300, 441)]
for fname in fnames:
    src = os.path.join(original_dataset_dir2, fname)
    dst = os.path.join(test_porebrik_dir, fname)
    shutil.copyfile(src, dst)