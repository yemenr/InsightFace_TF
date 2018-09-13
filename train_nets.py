import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import parse_function, distortion_parse_function
import os
# from nets.L_Resnet_E_IR import get_resnet
# from nets.L_Resnet_E_IR_GBN import get_resnet
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from losses.face_losses import arcface_loss, center_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test
import logging
import pdb
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=50,type=int, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, type=int, help='epoch to train the network')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--id_num_output', default=85742, type=int, help='the identity dataset class num')
    parser.add_argument('--seq_num_output', default=93979, type=int, help='the sequence dataset class num')
    parser.add_argument('--id_tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--seq_tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')                        
    parser.add_argument('--center_loss_alfa', type=float, help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--chief_loss_factor', type=float, help='chief loss factor.', default=0.96)
    parser.add_argument('--auxiliary_loss_factor', type=float, help='auxiliary loss factor.', default=0.04)
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', default=None, help='pretrained model')
    parser.add_argument('--devices', default='0', help='the ids of gpu devices')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename = 'train_out.log',level=logging.INFO, format = log_format)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='input', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=[], name='dropout_rate')
    
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    id_tfrecords_f = os.path.join(args.id_tfrecords_file_path, 'tran.tfrecords')
    seq_tfrecords_f = os.path.join(args.seq_tfrecords_file_path, 'tran.tfrecords')
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(id_tfrecords_f)
        dataset = dataset.map(distortion_parse_function)
        dataset = dataset.shuffle(buffer_size=args.buffer_size)
        dataset = dataset.batch(args.batch_size//2)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
    
        dataset1 = tf.data.TFRecordDataset(seq_tfrecords_f)
        dataset1 = dataset1.map(parse_function)
        dataset1 = dataset1.shuffle(buffer_size=args.buffer_size)
        dataset1 = dataset1.batch(args.batch_size//2)
        iterator1 = dataset1.make_initializable_iterator()
        next_element1 = iterator1.get_next()

    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        logging.info('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.id_num_output)
    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=True, keep_rate=dropout_rate)
    embedding_tensor = test_net.outputs
    
    # 3.2.a split logits and labels into identity dataset and sequence dataset
    idLogits, seqLogits = tf.split(logit,2,0)
    idLabels, seqLabels = tf.split(labels,2,0)
    
    # 3.3 define the cross entropy added LSA parts
    identity_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=idLogits, labels=idLabels))
    sequence_loss = -(tf.reduce_sum(tf.log(tf.nn.softmax(seqLogits)))/id_num_output
    chief_loss = identity_loss + sequence_loss
    
    # 3.3.a center loss
    logits_center_loss, _ = center_loss(logit, labels, args.center_loss_alfa, id_num_output+seq_num_output)
    auxiliary_loss = logits_center_loss
    
    # inference_loss_avg = tf.reduce_mean(inference_loss)
    # 3.4 define weight deacy losses
    # for var in tf.trainable_variables():
    #     print(var.name)
    # print('##########'*30)
    wd_loss = 0
    for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
    for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
    # for beta in tl.layers.get_variables_with_name('beta', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(beta)
    for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
    # for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(bias)

    # 3.5 total losses
    #total_loss = inference_loss + wd_loss
    total_loss = (chief_loss + wd_loss) * args.chief_loss_factor + auxiliary_loss * args.auxiliary_loss_factor
    # 3.6 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [p*val for val in args.lr_steps]
    print(lr_steps)
    logging.info(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    #modify mult-lr
    grads_and_vars_mult = []
    for grad, var in grads:
        if "spatial_trans" in var.op.name:
            grad *= 0.1
        grads_and_vars_mult.append((grad, var))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads_and_vars_mult, global_step=global_step)
    # train_op = opt.minimize(total_loss, global_step=global_step)
    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # 3.11 summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # # 3.11.1 add grad histogram op
    for grad, var in grads_and_vars_mult:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('chief_loss', chief_loss))
    summaries.append(tf.summary.scalar('auxiliary_loss', auxiliary_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())
    if args.pretrained_model:
        logging.info('Restoring pretrained model: %s' % args.pretrained_model)
        saver.restore(sess, args.pretrained_model)
    # restore_saver = tf.train.Saver()
    # restore_saver.restore(sess, '/home/aurora/workspaces2018/InsightFace_TF/output/ckpt/InsightFace_iter_1110000.ckpt')
    # 4 begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    # 4 begin iteration
    count = 0
    total_accuracy = {}

    for i in range(args.epoch):
        sess.run(iterator.initializer)
        sess.run(iterator1.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                images_train1, labels_train1 = sess.run(next_element1)
                train_data = np.concatenate((images_train, images_train1), axis=0)
                label_data = np.concatenate((labels_train, labels_train1), axis=0)
                feed_dict = {images: train_data, labels: label_data, dropout_rate: 0.4}
                feed_dict.update(net.all_drop)
                start = time.time()
                _, total_loss_val, chief_loss_val, auxiliary_loss_val, wd_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, chief_loss, auxiliary_loss, wd_loss, inc_op, acc],
                              feed_dict=feed_dict,
                              options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , chief loss is %.2f, auxiliary loss is %.2f, weight deacy '
                          'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, chief_loss_val, auxiliary_loss_val, wd_loss_val, acc_val, pre_sec))
                    logging.info('epoch %d, total_step %d, total loss is %.2f , chief loss is %.2f, auxiliary loss is %.2f, weight deacy '
                          'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, chief_loss_val, auxiliary_loss_val, wd_loss_val, acc_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: train_data, labels: label_data, dropout_rate: 0.4}
                    feed_dict.update(net.all_drop)
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)

                # validate
                if count > 0 and count % args.validate_interval == 0:
                    feed_dict_test ={dropout_rate: 1.0}
                    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                             input_placeholder=images)
                    print('test accuracy is: ', str(results[0]))
                    logging.info('test accuracy is: %s' % str(results[0]))
                    total_accuracy[str(count)] = results[0]
                    log_file.write('########'*10+'\n')
                    log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                    log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                    log_file.flush()
                    if max(results) > 0.996:
                        print('best accuracy is %.5f' % max(results))
                        logging.info('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
                        log_file.write('######Best Accuracy######'+'\n')
                        log_file.write(str(max(results))+'\n')
                        log_file.write(filename+'\n')

                        log_file.flush()
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                logging.info("End of epoch %d" % i)
                break
    log_file.close()
    log_file.write('\n')
