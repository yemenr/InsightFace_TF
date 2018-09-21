'''
import tensorflow as tf
import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

train_filename = 'train.tfrecords'
with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:  
    for i in range(len(images)):
        # read in image data by tf
        img_data = tf.gfile.FastGFile(images[i], 'rb').read()  # image data type is string
        label = labels[i]
        # get width and height of image
        image_shape = cv2.imread(images[i]).shape
        width = image_shape[1]
        height = image_shape[0]
        # create features
        feature = {'train/image': _bytes_feature(img_data),
                           'train/label': _int64_feature(label),  # label: integer from 0-N
                           'train/height': _int64_feature(height), 
                           'train/width': _int64_feature(width)}
        # create example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # serialize protocol buffer to string
        tfrecord_writer.write(example.SerializeToString())
 tfrecord_writer.close()
 '''
 
import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np
 
cwd='/home/cys/wcqk_data/data/faces/test_faces'
writer= tf.python_io.TFRecordWriter("test_faces.tfrecords") #要生成的文件
 
for index,className in enumerate(os.listdir(cwd)):
    class_path=os.path.join(cwd,className)
    for img_name in os.listdir(class_path): 
        img_path=os.path.join(class_path,img_name) #每一个图片的地址 
        img=Image.open(img_path)
        img= img.resize((128,128))
        image_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={            
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0-int(className)]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()

def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={                                           
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  #reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label

outDir = "output/"
filename_queue = tf.train.string_input_producer(["test_faces.tfrecords"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={                                       
                                       'image_raw' : tf.FixedLenFeature([], tf.string),
                                       'label': tf.FixedLenFeature([], tf.int64)
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['image_raw'], tf.uint8)
image = tf.reshape(image, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(outDir+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)    