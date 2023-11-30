# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified by Graphcore Ltd to use the KFold cross
# validation from sklearn.model_selection. The original file can be found
# here https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/UNet_Medical/data_loading/data_loader.py

from functools import partial
import multiprocessing
import numpy as np
import os
from PIL import Image, ImageSequence
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
data_root = "/home/tak/tiles_512/"
data_path = os.listdir(data_root)
tile_size = 512
csv_path = 'train_512.csv'

def test_data_load() :
    img_path = '/home/tak/experiments/MIL/examples/vision/unet_medical/tensorflow2/isbi-datasets/data/'

    img_list = os.listdir(os.path.join(img_path, 'images'))
    test_x = []
    for i, item in tqdm(enumerate(img_list)) :
        img = Image.open(os.path.join(os.path.join(img_path, 'images'),item)).convert('RGB')
        img = img.convert("L")
        resize_image = img.resize((512, 512))
        # test = np.array([np.array(p) for p in ImageSequence.Iterator(img)])
        image = np.array(resize_image)
        test_x.append(image)
    
    return test_x

def get_full_path(df, data_root, except_ext=True):
    return f"{data_root}/{df['folder']}/{df['slide']}/{df['file']}"

def make_dataset(dtype) :
    data_df = pd.read_csv(csv_path, index_col=0)
    input_images_path = data_df.apply(get_full_path, data_root=data_root, axis=1).tolist()
    has_mask = data_df["has_mask"].tolist()
    normal_mask = tf.zeros((tile_size, tile_size), dtype=tf.float32)

    dataset = []
    label_mask = []
    # print(has_mask[200000])
    for i,item in tqdm(enumerate(input_images_path)) :
        # if i == 300 : break

        if has_mask[i]:
            image = Image.open(item+'_T.png').convert('RGB')
            image = image.convert("L")
            mask = Image.open(item.replace("from_ts","from_tsm")+'_T_mask.png')

            image = np.array(image)
            mask = np.array(mask)
            dataset.append(image)
            label_mask.append(mask)
        else : 
            image = Image.open(item+'_N.png').convert('RGB')
            image = image.convert("L")
            mask = normal_mask

            image = np.array(image)
            mask = np.array(mask)
            dataset.append(image)
            label_mask.append(mask)

    dataset = np.array(dataset)
    label_mask = np.array(label_mask)
    return dataset, label_mask

def generate_numpy_data(args):
    nb_samples = 30
    X = np.random.uniform(size=(nb_samples, 572, 572)).astype(args.dtype)
    Y = np.random.uniform(size=(nb_samples, 388, 388, args.nb_classes)).astype(args.dtype)
    X_test = np.random.uniform(size=(nb_samples, 572, 572)).astype(args.dtype)
    return X, Y, X_test


def get_images_labels(args):
    images_path = os.path.join(args.data_dir, "train-volume.tif")
    masks_path = os.path.join(args.data_dir, "train-labels.tif")
    test_images_path = os.path.join(args.data_dir, "test-volume.tif")
    inputs = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(images_path))])
    labels = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(masks_path))])
    test_inputs = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(test_images_path))])
    return inputs, labels, test_inputs


def _normalize_inputs(inputs, dtype):
    """Normalize inputs"""
    inputs = tf.expand_dims(tf.cast(inputs, dtype), -1)
    # Center around zero
    inputs = tf.divide(inputs, 127.5) - 1
    # Resize to match output size
    inputs = tf.image.resize(inputs, (388, 388))

    return tf.image.resize_with_crop_or_pad(inputs, 572, 572)


def _normalize_labels(labels):
    """Normalize labels"""
    labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
    labels = tf.divide(labels, 255)

    # Resize to match output size
    labels = tf.image.resize(labels, (388, 388))
    labels = tf.image.resize_with_crop_or_pad(labels, 572, 572)

    cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))
    return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)


def data_augmentation(inputs, labels):
    # Horizontal flip
    h_flip = tf.random.uniform([]) > 0.5
    inputs = tf.cond(pred=h_flip, true_fn=lambda: tf.image.flip_left_right(inputs), false_fn=lambda: inputs)
    labels = tf.cond(pred=h_flip, true_fn=lambda: tf.image.flip_left_right(labels), false_fn=lambda: labels)

    # Vertical flip
    v_flip = tf.random.uniform([]) > 0.5
    inputs = tf.cond(pred=v_flip, true_fn=lambda: tf.image.flip_up_down(inputs), false_fn=lambda: inputs)
    labels = tf.cond(pred=v_flip, true_fn=lambda: tf.image.flip_up_down(labels), false_fn=lambda: labels)

    # Prepare for batched transforms
    inputs = tf.expand_dims(inputs, 0)
    labels = tf.expand_dims(labels, 0)

    # Random crop and resize
    left = tf.random.uniform([]) * 0.3
    right = 1 - tf.random.uniform([]) * 0.3
    top = tf.random.uniform([]) * 0.3
    bottom = 1 - tf.random.uniform([]) * 0.3

    inputs = tf.image.crop_and_resize(inputs, [[top, left, bottom, right]], [0], (572, 572))
    labels = tf.image.crop_and_resize(labels, [[top, left, bottom, right]], [0], (572, 572))
    inputs = tf.squeeze(inputs, 0)
    labels = tf.squeeze(labels, 0)
    # random brightness and keep values in range
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.clip_by_value(inputs, clip_value_min=-1, clip_value_max=1)
    return inputs, labels


def preprocess_fn(inputs, labels, dtype, augment=False):
    inputs = _normalize_inputs(inputs, dtype)
    labels = _normalize_labels(labels)
    if augment:
        inputs, labels = data_augmentation(inputs, labels)

    # Bring back labels to network's output size and remove interpolation artifacts
    labels = tf.image.resize_with_crop_or_pad(labels, target_width=388, target_height=388)
    cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

    # cast inputs and labels to given dtype
    inputs = tf.cast(inputs, dtype)
    labels = tf.cast(labels, dtype)
    return inputs, labels

def _normalize_labels_custom(labels):
    """Normalize labels"""
    labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
    # labels = tf.divide(labels, 255)

    # Resize to match output size
    labels = tf.image.resize(labels, (388, 388))
    labels = tf.image.resize_with_crop_or_pad(labels, 572, 572)

    # cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    cond = tf.greater(labels, tf.zeros(tf.shape(input=labels)))
    # labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))
    labels = tf.where(cond, tf.ones(tf.shape(input=labels)), tf.zeros(tf.shape(input=labels)))
    return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)

def preprocess_fn_custom(inputs, labels, dtype, augment=False):
    inputs = _normalize_inputs(inputs, dtype)
    labels = _normalize_labels_custom(labels)
    if augment:
        inputs, labels = data_augmentation(inputs, labels)

    # Bring back labels to network's output size and remove interpolation artifacts
    labels = tf.image.resize_with_crop_or_pad(labels, target_width=388, target_height=388)
    # cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    # labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))
    cond = tf.greater(labels, tf.zeros(tf.shape(input=labels)))
    labels = tf.where(cond, tf.ones(tf.shape(input=labels)), tf.zeros(tf.shape(input=labels)))
    # cast inputs and labels to given dtype
    # print(labels)
    inputs = tf.cast(inputs, dtype)
    labels = tf.cast(labels, dtype)
    return inputs, labels

def tf_fit_dataset_custom(args, inputs, labels):
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    ds = ds.shuffle(inputs.shape[0], seed=args.seed)
    ds = ds.repeat()

    if not args.host_generated_data:
        ds = ds.map(
            partial(preprocess_fn_custom, dtype=args.dtype, augment=args.augment),
            num_parallel_calls=min(32, multiprocessing.cpu_count()),
        )
    ds = ds.batch(args.micro_batch_size, drop_remainder=True)
    if args.use_prefetch:
        ds = ds.prefetch(args.steps_per_execution)
    return ds

def tf_eval_dataset_custom(args, X_eval, y_eval):
    ds = tf.data.Dataset.from_tensor_slices((X_eval, y_eval))
    ds = ds.repeat(count=args.gradient_accumulation_count // len(X_eval) + 1)
    if not args.host_generated_data:
        ds = ds.map(partial(preprocess_fn_custom, dtype=args.dtype), num_parallel_calls=min(32, multiprocessing.cpu_count()))
    ds = ds.batch(args.micro_batch_size, drop_remainder=True)

    return ds

def tf_fit_dataset(args, inputs, labels):
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    ds = ds.shuffle(inputs.shape[0], seed=args.seed)
    ds = ds.repeat()

    if not args.host_generated_data:
        ds = ds.map(
            partial(preprocess_fn, dtype=args.dtype, augment=args.augment),
            num_parallel_calls=min(32, multiprocessing.cpu_count()),
        )
    ds = ds.batch(args.micro_batch_size, drop_remainder=True)
    if args.use_prefetch:
        ds = ds.prefetch(args.steps_per_execution)
    return ds


def tf_eval_dataset(args, X_eval, y_eval):
    ds = tf.data.Dataset.from_tensor_slices((X_eval, y_eval))
    ds = ds.repeat(count=args.gradient_accumulation_count // len(X_eval) + 1)
    if not args.host_generated_data:
        ds = ds.map(partial(preprocess_fn, dtype=args.dtype), num_parallel_calls=min(32, multiprocessing.cpu_count()))
    ds = ds.batch(args.micro_batch_size, drop_remainder=True)

    return ds


def predict_data_set(args, X):
    ds = tf.data.Dataset.from_tensor_slices((X))
    ds = ds.repeat()
    if not args.host_generated_data:
        ds = ds.map(partial(_normalize_inputs, dtype=args.dtype))
    ds = ds.batch(args.micro_batch_size, drop_remainder=True)
    if args.use_prefetch:
        ds = ds.prefetch(args.steps_per_execution)
    return ds