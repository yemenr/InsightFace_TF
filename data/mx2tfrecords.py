import mxnet as mx
import argparse
import PIL.Image
import io
import numpy as np
import cv2
import tensorflow as tf
import os
from scipy import misc
import pdb
import math

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='../datasets/faces_ms1m_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='../datasets/faces_ms1m_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    args = parser.parse_args()
    return args


def mx2tfrecords_old(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        encoded_jpg_io = io.BytesIO(img)
        image = PIL.Image.open(encoded_jpg_io)
        np_img = np.array(image)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        img_raw = img.tobytes()
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        if (type(header.label)==float) :
            label = int(header.label)
        else:
            label = int(header.label[0])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()

def mx2tfrecords_new(imgidx, imgrec, args):
    #output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    #writer = tf.python_io.TFRecordWriter(output_path)
    writer = None
    labelSet = set()
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        if (type(header.label)==float) :
            label = int(header.label)
        else:
            label = int(header.label[0])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        if label not in labelSet:
            if writer is not None:
                writer.close()
            labelSet.add(label)
            output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords'+str(label))
            writer = tf.python_io.TFRecordWriter(output_path)
        if writer is not None:
            writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    if writer is not None:
        writer.close()



def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
    
def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def _rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """Rotate the given image with the given rotation degree and crop for the black edges if necessary
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.

    Returns:
        A rotated image.
    """
  
    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:
        image = tf.contrib.image.rotate(image, math.radians(rotation_degree), interpolation='BILINEAR')
      
        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = _largest_rotated_rect(output_height, output_width, math.radians(rotation_degree))
            print(float(lrr_height)/output_height)
            resized_image = tf.image.central_crop(image, float(lrr_height)/output_height)    
            image = tf.image.resize_images(resized_image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            #image = tf.image.resize_image_with_crop_or_pad(resized_image, output_height, output_width)
    return image    

def image_enhance(srcImg, output_height, output_width):
    angle = np.random.uniform(low=-30.0, high=30.0)
    #img = _rotate_and_crop(srcImg, output_height, output_width, angle, True)
    
    img = tf.image.random_flip_left_right(srcImg)
    
    #img = tf.image.random_brightness(img, 0.2)
    
    #img = tf.image.random_contrast(img, 0.5, 1.2)    

    return img
    
def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    #img = tf.concat([b, g, r], axis=-1)
    img = tf.concat([r, g, b], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    #img = tf.subtract(img, 127.5)
    #img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int32)
    
    #realLabel = tf.cond(tf.less(label,0), lambda: tf.subtract(85741,label),lambda: label)
    #realLabel = tf.cond(tf.less(label,0), lambda: tf.subtract(63,label),lambda: label)
    #realLabel = tf.cast(realLabel, tf.int64)
    realLabel = tf.cast(label, tf.int64)
    
    return img, realLabel

def distortion_parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    #img = tf.concat([b, g, r], axis=-1)
    img = tf.concat([r, g, b], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    #img = tf.subtract(img, 127.5)
    #img = tf.multiply(img,  0.0078125)
    
    # image enhancement
    img = image_enhance(img, 112, 112)
    
    label = tf.cast(features['label'], tf.int32)
    
    #realLabel = tf.cond(tf.less(label,0), lambda: tf.subtract(85741,label),lambda: label)    
    realLabel = tf.cond(tf.less(label,0), lambda: tf.subtract(63,label),lambda: label)
    realLabel = tf.cast(realLabel, tf.int64)
    
    return img, realLabel

def generator():
    while True:
        # Sample the labels that will compose the batch
        labels = np.random.choice(range(num_labels),
                                  num_classes_per_batch,
                                  replace=False)
        for label in labels:
            for _ in range(num_images_per_class):
                yield label    
    
if __name__ == '__main__':
    # # define parameters
    id2range = {}
    data_shape = (3, 112, 112)
    args = parse_args()
    imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)
    print('id2range', len(id2range))
    
    # generate tfrecords
    #mx2tfrecords(imgidx, imgrec, args)
    mx2tfrecords_new(imgidx, imgrec, args)

    #load tfrecords test
    
    #config = tf.ConfigProto(allow_soft_placement=True)
    #sess = tf.Session(config=config)
    ## training datasets api config    
    #tfNames = [os.path.join(args.tfrecords_file_path, 'tran.tfrecords'+str(-k)) for k in range(1,298)]

    #dataset1 = tf.data.TFRecordDataset(tfNames)
    #dataset1 = dataset1.map(parse_function)
    #dataset1 = dataset1.shuffle(buffer_size=30000)
    #dataset1 = dataset1.batch(32)
    #iterator1 = dataset1.make_initializable_iterator()
    #next_element1 = iterator1.get_next()

    #per_class_datasets = [tf.data.TFRecordDataset(f).repeat(2).map(parse_function) for f in tfNames]

    #num_labels = 297
    #num_classes_per_batch = 4
    #num_images_per_class = 8
    #pdb.set_trace()
    
    #choice_dataset = tf.data.Dataset.from_generator(generator, tf.int64)
    #dataset = tf.contrib.data.choose_from_datasets(per_class_datasets, choice_dataset)
    #batch_size = num_classes_per_batch * num_images_per_class
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(None)
    #iterator = dataset.make_initializable_iterator()
    #next_element = iterator.get_next()
    # begin iteration
    #for i in range(2):
    #    sess.run(iterator.initializer)
    #    sess.run(iterator1.initializer)
    #    while True:
    #        try:
    #            images, labels = sess.run(next_element)
    #            #images1, labels1 = sess.run(next_element1)
    #            print("1: ", labels)
    #            #print("2: ", labels1)
    #            #cv2.imshow('test', images[1, ...])
    #            #cv2.waitKey(0)
    #        except tf.errors.OutOfRangeError:
    #           print("End of dataset")
    #            break
