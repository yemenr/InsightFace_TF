import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import parse_function, distortion_parse_function
import os
from nets.L_Resnet_E_IR_MGPU import get_resnet
from losses.face_losses import arcface_loss, center_loss, single_dsa_loss, multiple_dsa_loss
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
    parser.add_argument('--lr_steps', type=str, default='', help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=8e-4, type=float, help='learning alg momentum')
    #parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'survellance'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['survellance'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--id_num_output', default=85742, type=int, help='the identity dataset class num')
    parser.add_argument('--seq_num_output', default=93979, type=int, help='the sequence dataset class num')
    parser.add_argument('--id_tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--seq_tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')                        
    parser.add_argument('--center_loss_alfa', type=float, help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--auxiliary_loss_factor', type=float, help='auxiliary loss factor.', default=1)
    parser.add_argument('--norm_loss_factor', type=float, help='norm loss factor.', default=0)
    parser.add_argument('--sequence_loss_factor', type=float, help='sequence loss factor.', default=1)
    parser.add_argument('--dsa_param', default=[0.5, 2, 1, 0.005], help='[dsa_lambda, dsa_alpha, dsa_beta, dsa_p]')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=100000, type=int, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=5000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, type=int, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to show information')
    parser.add_argument('--num_gpus', default=2, help='the num of gpus')
    parser.add_argument('--tower_name', default='tower', help='tower name')
    parser.add_argument('--pretrained_model', default=None, help='pretrained model')
    parser.add_argument('--devices', default='0', help='the ids of gpu devices')
    parser.add_argument('--log_file_name', default='train_out.log', help='the ids of gpu devices')
    parser.add_argument('--dataset_type', default='multiple', help='single dataset or multiple dataset')
    parser.add_argument('--lsr', action='store_true', help='add LSR item')
    parser.add_argument('--aux_loss_type', default=None, help='None | center | dsa loss')
    args = parser.parse_args()
    return args


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads   
    
if __name__ == '__main__':
    args = get_parser()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename = args.log_file_name,level=logging.INFO, format = log_format)

    # 1. define global parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    id_images = tf.placeholder(name='id_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    seq_images = tf.placeholder(name='seq_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    images_test = tf.placeholder(name='input', shape=[None, *args.image_size, 3], dtype=tf.float32)
    id_labels = tf.placeholder(name='id_labels', shape=[None, ], dtype=tf.int64)
    seq_labels = tf.placeholder(name='seq_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=[], name='dropout_rate') 
    
    # splits input to different gpu
    id_images_s = tf.split(id_images, num_or_size_splits=args.num_gpus, axis=0)
    seq_images_s = tf.split(seq_images, num_or_size_splits=args.num_gpus, axis=0)
    id_labels_s = tf.split(id_labels, num_or_size_splits=args.num_gpus, axis=0)
    seq_labels_s = tf.split(seq_labels, num_or_size_splits=args.num_gpus, axis=0)
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
    # 3.2 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [p*val for val in args.lr_steps]
    print('learning rate steps: ', lr_steps)
    logging.info(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001],name='lr_schedule')
    # 3.3 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)

    # Calculate the gradients for each model tower.
    tower_grads = []
    tl.layers.set_name_reuse(True)
    loss_dict = {}
    drop_dict = {}
    loss_keys = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(args.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (args.tower_name, i)) as scope:
            concat_images = tf.concat([id_images_s[i], seq_images_s[i]],0)
            concat_labels = tf.concat([id_labels_s[i], seq_labels_s[i]],0)
            net = get_resnet(concat_images, args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
            
            with tf.variable_scope("logits"):
                logit = arcface_loss(embedding=net.outputs, labels=concat_labels, w_init=w_init_method, out_num=args.id_num_output)
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            
            # 3.2.a split logits and labels into identity dataset and sequence dataset
            idLogits, seqLogits = tf.split(logit,2,0)
    
            # define the cross entropy
            identity_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=idLogits, labels=id_labels_s[i]))
            sequence_loss = -tf.reduce_mean(tf.log(tf.nn.softmax(seqLogits))) # warning
            chief_loss = identity_loss*args.identity_loss_factor + sequence_loss*(1-args.identity_loss_factor)
            # center loss & dsa loss
            logits_center_loss, _ = center_loss(net.outputs, concat_labels, args.center_loss_alfa, args.id_num_output+args.seq_num_output)
            #feature_dsa_loss, _ = dsa_loss(net.outputs, concat_labels, args.center_loss_alfa, args.id_num_output, args.seq_num_output, args.dsa_param, args.batch_size)
            auxiliary_loss = logits_center_loss
            # define weight deacy losses
            wd_loss = 0
            for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
            for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
            for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
            for gamma in tl.layers.get_variables_with_name('gamma', True, True):
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
            for alphas in tl.layers.get_variables_with_name('alphas', True, True):
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
            #total_loss = inference_loss + wd_loss
            total_loss = chief_loss * args.chief_loss_factor + auxiliary_loss * (1 - args.chief_loss_factor) + wd_loss*args.norm_loss_factor

            loss_dict[('chief_loss_%s_%d' % ('gpu', i))] = chief_loss
            loss_keys.append(('chief_loss_%s_%d' % ('gpu', i)))
            loss_dict[('auxiliary_loss_%s_%d' % ('gpu', i))] = auxiliary_loss
            loss_keys.append(('auxiliary_loss_%s_%d' % ('gpu', i)))
            loss_dict[('wd_loss_%s_%d' % ('gpu', i))] = wd_loss
            loss_keys.append(('wd_loss_%s_%d' % ('gpu', i)))
            loss_dict[('total_loss_%s_%d' % ('gpu', i))] = total_loss
            loss_keys.append(('total_loss_%s_%d' % ('gpu', i)))
            grads = opt.compute_gradients(total_loss)
            tower_grads.append(grads)
            if i == 0:
                test_net = get_resnet(images_test, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
                embedding_tensor = test_net.outputs
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                pred = tf.nn.softmax(idLogits)
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), id_labels_s[i]), dtype=tf.float32))

    grads = average_gradients(tower_grads)
    #modify mult-lr
    grads_and_vars_mult = []
    for grad, var in grads:
        if "spatial_trans" in var.op.name:
            grad *= 0.1
        grads_and_vars_mult.append((grad, var))
        
    with tf.control_dependencies(update_ops):
        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads_and_vars_mult, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # add grad histogram op
    for grad, var in grads_and_vars_mult:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # add loss summary
    for keys, val in loss_dict.items():
        summaries.append(tf.summary.scalar(keys, val))
    # add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    # init all variables
    sess.run(tf.global_variables_initializer())
    if args.pretrained_model:
        logging.info('Restoring pretrained model: %s' % args.pretrained_model)
        saver.restore(sess, args.pretrained_model)

    sess.graph.finalize()
    # begin iteration
    count = 0
    for i in range(args.epoch):
        sess.run(iterator.initializer)
        sess.run(iterator1.initializer)
        while True:
            try:
                id_images_train, id_labels_train = sess.run(next_element)
                seq_images_train, seq_labels_train = sess.run(next_element1)
                
                feed_dict = {id_images: id_images_train, id_labels: id_labels_train, seq_images: seq_images_train, seq_labels: seq_labels_train, dropout_rate: 0.4}
                start = time.time()
                _, _, chief_loss_val_gpu_1, auxiliary_loss_val_gpu_1, wd_loss_val_gpu_1, total_loss_gpu_1, chief_loss_val_gpu_2, \
                auxiliary_loss_val_gpu_2, wd_loss_val_gpu_2, total_loss_gpu_2, acc_val = sess.run([train_op, inc_op, loss_dict[loss_keys[0]],
                                                                         loss_dict[loss_keys[1]],
                                                                         loss_dict[loss_keys[2]],
                                                                         loss_dict[loss_keys[3]],
                                                                         loss_dict[loss_keys[4]],
                                                                         loss_dict[loss_keys[5]],
                                                                         loss_dict[loss_keys[6]],
                                                                         loss_dict[loss_keys[7]],acc],
                                                                         feed_dict=feed_dict)
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss: [%.2f, %.2f], chief loss: [%.2f, %.2f], auxiliary loss: [%.2f, %.2f], weight deacy '
                          'loss: [%.2f, %.2f], training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_gpu_1, total_loss_gpu_2, chief_loss_val_gpu_1, chief_loss_val_gpu_2, auxiliary_loss_val_gpu_1, auxiliary_loss_val_gpu_2,
                           wd_loss_val_gpu_1, wd_loss_val_gpu_2, acc_val, pre_sec))
                    logging.info('epoch %d, total_step %d, total loss: [%.2f, %.2f], chief loss: [%.2f, %.2f], auxiliary loss: [%.2f, %.2f], weight deacy '
                          'loss: [%.2f, %.2f], training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_gpu_1, total_loss_gpu_2, chief_loss_val_gpu_1, chief_loss_val_gpu_2, auxiliary_loss_val_gpu_1, auxiliary_loss_val_gpu_2,
                           wd_loss_val_gpu_1, wd_loss_val_gpu_2, acc_val, pre_sec))       
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)
                # # validate
                if count >= 0 and count % args.validate_interval == 0:
                    feed_dict_test ={dropout_rate: 1.0}
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=args.batch_size//args.num_gpus, feed_dict=feed_dict_test,
                             input_placeholder=images_test)
                    logging.info("lfw test accuracy is: %.5f" % (results[0]))
                    if max(results) > 0.995:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        logging.info('best accuracy is %.5f' % max(results))
                        saver.save(sess, filename)
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                logging.info("End of epoch %d" % i)
                break
