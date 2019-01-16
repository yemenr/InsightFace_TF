import tensorflow as tf
import argparse
from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss
from nets.mxnet_pretrain import get_resnet
import tensorlayer as tl
from verification import ver_test


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file', default='/home/ubuntu/camel/models/ckpt_model_d/InsightFace_iter_best_',
                       type=str, help='the ckpt file path')
    #parser.add_argument('--eval_datasets', default=['survellance', 'lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['surveillance','lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='/home/ubuntu/camel/workspace/data/faces_emore', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=16, help='batch size to train network')
    parser.add_argument('--ckpt_index_list',
                        default=['710000.ckpt'], help='ckpt file indexes')
    parser.add_argument('--weight_file', default=None, help='mxnet r100 weight file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)

    images = tf.placeholder(name='input', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, w_init=w_init_method, trainable=False, keep_rate=dropout_rate, weight_file=args.weight_file)
    embedding_tensor = net
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())    #if restore by saver
    sess.run(tf.local_variables_initializer())
    #saver = tf.train.Saver()

    #result_index = []
    #for file_index in args.ckpt_index_list:
    #   feed_dict_test = {}
    #   path = args.ckpt_file + file_index
    #   saver.restore(sess, path)
    #   print('ckpt file %s restored!' % file_index)
    #   feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
    #   feed_dict_test[dropout_rate] = 1.0
    #   results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
    #                      embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
    #                      input_placeholder=images)
    #   result_index.append(results)
    #print(result_index)

    feed_dict_test = {}
    feed_dict_test[dropout_rate] = 1.0
    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                       embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test, input_placeholder=images)
    print(results)
    