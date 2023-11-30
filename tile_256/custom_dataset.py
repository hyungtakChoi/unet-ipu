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

data_root = "/home/tak/experiments/MIL/examples/vision/unet_medical/tensorflow2/dataset/tiles"
# data_root = "/home/tak/tiles_512/"
data_path = os.listdir(data_root)
tile_size = 256
csv_path = 'train.csv'

def make_csv_camelyon16_tiles(data_root='/home/tak/tiles_512/', output_path='train_512.csv'):
    import glob

    normal_df = pd.DataFrame()
    tumor_df = pd.DataFrame()

    # Normal tiles
    normal_ns_tiles = glob.glob(data_root+'np_from_ns/*/*.png') # '_N.png'
    normal_ts_tiles = glob.glob(data_root+'np_from_ts/*/*.png') # '_N.png'

    normal_df['file'] = normal_ns_tiles + normal_ts_tiles
    print(normal_df['file'].str.split('/', expand=True))
    # normal_df[['0','1','2','3','4','5','6','7','8','9','10','folder','slide','file']] = normal_df['file'].str.split('/', expand=True)
    normal_df[['0','1','2','3','folder','slide','file']] = normal_df['file'].str.split('/', expand=True)
    normal_df['file'] = normal_df['file'].str.split('_N.').str[0]
    normal_df = normal_df[['folder','slide','file']]
    normal_df['has_mask'] = False

    # Tumor tiles
    tumor_ts_tiles = glob.glob(data_root+'tp_from_ts/*/*.png') # '_T.png'

    tumor_df['file'] = tumor_ts_tiles
    # tumor_df[['0','1','2','3','4','5','6','7','8','9','10','folder','slide','file']] = tumor_df['file'].str.split('/', expand=True)
    tumor_df[['0','1','2','3','folder','slide','file']] = tumor_df['file'].str.split('/', expand=True)
    tumor_df['file'] = tumor_df['file'].str.split('_T.').str[0]
    tumor_df = tumor_df[['folder','slide','file']]
    tumor_df['has_mask'] = True

    # concat both
    total_df = pd.concat([normal_df, tumor_df])
    total_df = total_df.reset_index(drop=True)
    total_df.to_csv(output_path)

def get_full_path(df, data_root, except_ext=True):
    return f"{data_root}/{df['folder']}/{df['slide']}/{df['file']}"

def make_dataset(dtype) :
    data_df = pd.read_csv(csv_path, index_col=0)
    input_images_path = data_df.apply(get_full_path, data_root=data_root, axis=1).tolist()
    has_mask = data_df["has_mask"].tolist()
    normal_mask = tf.zeros((tile_size, tile_size), dtype=tf.float32)

    dataset = []
    label_mask = []
    print(has_mask.count(True))
    print(has_mask.count(False))

    for i,item in tqdm(enumerate(input_images_path)) :
        # if i == 1 : break
        if has_mask[i]:
            image = Image.open(item+'_T.png').convert('RGB')
            image = image.convert("L")
            # print(f'image : {image}')
            # input = np.array([np.array(p) for p in ImageSequence.Iterator(image)])
            mask = Image.open(item.replace("from_ts","from_tsm")+'_T_mask.png')
            # mask = np.array([np.array(p) for p in ImageSequence.Iterator(mask)])
            # mask = torch.from_numpy(np.array(mask)).float()
            # print(f'mask : {mask}')
            image = np.array(image).astype(int)
            mask = np.array(mask).astype(int)
            dataset.append(image)
            label_mask.append(mask)
        else : 
            image = Image.open(item+'_N.png').convert('RGB')
            image = image.convert("L")
            # print(f'image : {image}')
            # input = np.array([np.array(p) for p in ImageSequence.Iterator(image)])
            mask = normal_mask
            # mask = np.array([np.array(p) for p in ImageSequence.Iterator(mask)])
            # print(f'mask : {mask}')
            image = np.array(image).astype(int)
            mask = np.array(mask).astype(int)
            dataset.append(image)
            label_mask.append(mask)

    return dataset, label_mask


def dataset_preprocess(img_list, img_path) :
    data_list = []
    for i, item in enumerate(img_list) :
        data_path = os.path.join(img_path, item)
        # inputs = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(data_path))])
        image = Image.open(data_path)
        data_list.append(inputs)

    return data_list


def my_data(args) :
    img_list = os.listdir(os.path.join(data_root, 'images'))
    train_image_list = [file for file in img_list if file.startswith("normal_")]
    test_image_list = [file for file in img_list if file.startswith("test_")]
    mask_list = os.listdir(os.path.join(data_root, 'masks'))
    train_mask_list = [file for file in mask_list if file.startswith("normal_")]

    inputs = dataset_preprocess(train_image_list, os.path.join(data_root, 'images'))
    labels = dataset_preprocess(train_mask_list, os.path.join(data_root, 'images'))
    test_inputs = dataset_preprocess(test_image_list, os.path.join(data_root, 'masks'))

    return inputs, labels, test_inputs


def generate_numpy_data(args):
    nb_samples = 30
    X = np.random.uniform(size=(nb_samples, tile_size, tile_size, 3)).astype(args.dtype)
    Y = np.random.uniform(size=(nb_samples, tile_size, tile_size, args.nb_classes)).astype(args.dtype)
    X_test = np.random.uniform(size=(nb_samples, tile_size, tile_size)).astype(args.dtype)
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
    # # Center around zero
    inputs = tf.divide(inputs, 127.5) - 1
    # # Resize to match output size
    # inputs = tf.image.resize(inputs, (256, 256))
    # input_image = tf.cast(inputs, tf.float32) / 255.0
    return inputs
    # return tf.image.resize_with_crop_or_pad(inputs, 512, 512)


def _normalize_labels(labels):
    """Normalize labels"""
    labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
    labels = tf.divide(labels, 255)

    # Resize to match output size
    # labels = tf.image.resize(labels, (256, 256))
    # labels = tf.image.resize_with_crop_or_pad(labels, 256, 256)

    # cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    # labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))
    cond = tf.greater(labels, tf.zeros(tf.shape(input=labels)))
    labels = tf.where(cond, tf.ones(tf.shape(input=labels)), tf.zeros(tf.shape(input=labels)))
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

    inputs = tf.image.crop_and_resize(inputs, [[top, left, bottom, right]], [0], (tile_size, tile_size))
    labels = tf.image.crop_and_resize(labels, [[top, left, bottom, right]], [0], (tile_size, tile_size))
    inputs = tf.squeeze(inputs, 0)
    labels = tf.squeeze(labels, 0)
    # random brightness and keep values in range
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.clip_by_value(inputs, clip_value_min=-1, clip_value_max=1)
    return inputs, labels


def preprocess_fn(inputs, labels, dtype, augment=False):
    inputs = _normalize_inputs(inputs, dtype)
    labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
    labels = _normalize_labels(labels)
    if augment:
        inputs, labels = data_augmentation(inputs, labels)

    # Bring back labels to network's output size and remove interpolation artifacts
    labels = tf.image.resize_with_crop_or_pad(labels, target_width=tile_size, target_height=tile_size)
    cond = tf.greater(labels, tf.zeros(tf.shape(input=labels)))
    labels = tf.where(cond, tf.ones(tf.shape(input=labels)), tf.zeros(tf.shape(input=labels)))
    # cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    # labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

    # cast inputs and labels to given dtype
    inputs = tf.cast(inputs, dtype)
    labels = tf.cast(labels, dtype)
    return inputs, labels


def tf_fit_dataset(args, inputs, labels):
    # print(type(inputs))
    # print(type(labels))
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    # print(ds)
    # ds = ds.shuffle(inputs.shape[0], seed=args.seed)
    # print(len(inputs))
    ds = ds.shuffle(len(inputs), seed=args.seed)
    ds = ds.repeat()

    if not args.host_generated_data:
        ds = ds.map(
            partial(preprocess_fn, dtype=args.dtype, augment=args.augment),
            num_parallel_calls=min(32, multiprocessing.cpu_count()),
        )
    ds = ds.batch(args.micro_batch_size, drop_remainder=True)
    if args.use_prefetch:
        ds = ds.prefetch(args.steps_per_execution)

    # print(ds)
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

def test_data_load() :
    img_path = '/home/tak/experiments/MIL/examples/vision/unet_medical/tensorflow2/isbi-datasets/data/'

    img_list = os.listdir(os.path.join(img_path, 'images'))
    test_x = []
    for i, item in tqdm(enumerate(img_list)) :
        img = Image.open(os.path.join(os.path.join(img_path, 'images'),item)).convert('RGB')
        img = img.convert("L")
        resize_image = img.resize((256, 256))
        # test = np.array([np.array(p) for p in ImageSequence.Iterator(img)])
        image = np.array(resize_image)
        test_x.append(image)
    
    return test_x

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(input, label):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'input': _int64_feature(input),
      'label': _int64_feature(label),
  }

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def tf_serialize_example(input, label):
  tf_string = tf.py_function(
    serialize_example,
    (input, label),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar.


def make_tfrecord(input, label, filename) :
    ds = tf.data.Dataset.from_tensor_slices((input, label))
    serialized_features_dataset = ds.map(tf_serialize_example)

    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)
