import os
import pandas as pd
import numpy as np
import tensorflow as tf

import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


def balance(df, n, working_dir, img_size):
    df = df.copy()
    aug_dir = os.path.join(working_dir, 'Treenuts_Aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in classes:
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)
    total = 0
    gen = ImageDataGenerator(horizontal_flip = True, rotation_range = 20, width_shift_range = .2, height_shift_range = .2, zoom_range = .2)
    groups = df.groupby('labels')
    for label in classes:
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < n:
            aug_img_count = 0
            delta = n - sample_count
            target_dir = os.path.join(aug_dir, label)
            aug_gen = gen.flow_from_dataframe(group, x_col = 'filepaths', y_col = None, target_size = img_size, class_mode = None, batch_size = 1, shuffle = False, save_to_dir = target_dir, save_prefix = 'aug-', color_mode = 'rgb', save_format = 'jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
            total += aug_img_count
    aug_fpaths = []
    aug_labels = []
    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name = 'filepaths')
    Lseries = pd.Series(aug_labels, name = 'labels')
    aug_df = pd.concat([Fseries, Lseries], axis = 1)
    df = pd.concat([df, aug_df], axis = 0).reset_index(drop = True)
    return df


with tf.device('cpu:0'):
    working_dir = os.path.dirname(__file__)

    data_csv = pd.read_csv(os.path.join(working_dir, 'tree nuts.csv'))
    groups = data_csv.groupby('data set')
    train_set = groups.get_group('train')
    test_set = groups.get_group('test')
    valid_set = groups.get_group('valid')

    train_set = train_set.drop(['data set', 'class index'], axis = 1)
    test_set = test_set.drop(['data set', 'class index'], axis = 1)
    valid_set = valid_set.drop(['data set', 'class index'], axis = 1)

    train_set['filepaths'] = working_dir + '/' + train_set['filepaths']
    test_set['filepaths'] = working_dir + '/' + test_set['filepaths']
    valid_set['filepaths'] = working_dir + '/' + valid_set['filepaths']

    classes = sorted(list(train_set['labels'].unique()))

    n = 200
    img_size = (224, 224)
    train_df = balance(train_set, n, working_dir, img_size)

    batch_size = 32

    trgen = ImageDataGenerator(horizontal_flip = True, rotation_range = 20, width_shift_range = .2, height_shift_range = .2, zoom_range = .2)
    t_and_v_gen = ImageDataGenerator()

    train_gen = trgen.flow_from_dataframe(train_df, x_col = 'filepaths', y_col = 'labels', target_size = img_size, class_mode = 'categorical', color_mode = 'rgb', shuffle = True, batch_size = batch_size)

    valid_gen = t_and_v_gen.flow_from_dataframe(valid_set, x_col = 'filepaths', y_col = 'labels', target_size = img_size, class_mode = 'categorical', color_mode = 'rgb', shuffle = False, batch_size = batch_size)

    test_gen = t_and_v_gen.flow_from_dataframe(test_set, x_col = 'filepaths', y_col = 'labels', target_size = img_size, class_mode = 'categorical', color_mode = 'rgb', shuffle = False, batch_size = batch_size)

    class_count = len(classes)
    img_shape = (img_size[0], img_size[1], 3)
    model_name = 'MobileNetV2'

    base_model = tf.keras.applications.MobileNetV2(include_top = False, weights = 'imagenet', input_shape = img_shape, pooling = 'max')
    base_model.trainable = True
    x = base_model.output
    x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016), activity_regularizer = regularizers.l1(0.006), bias_regularizer = regularizers.l1(0.006), activation = 'relu')(x)
    x = Dropout(rate = .4, seed = 123)(x)
    output = Dense(class_count, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = output)
    lr = 0.001
    model.compile(Adamax(learning_rate = lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x = train_gen, epochs = 10, verbose = 1, validation_data = valid_gen, validation_steps = None, shuffle = False)

    model.evaluate(test_gen)