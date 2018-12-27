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
    images = tf.placeholder(name='input', shape=[None, *args.image_size, 3], dtype=tf.float32)
    images_test = tf.placeholder(name='input', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=[], name='dropout_rate') 
    
    # splits input to different gpu
    images_s = tf.split(images, num_or_size_splits=args.num_gpus, axis=0)
    labels_s = tf.split(labels, num_or_size_splits=args.num_gpus, axis=0)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    id_tfrecords_f = os.path.join(args.id_tfrecords_file_path, 'tran.tfrecords')
    seq_tfrecords_f = os.path.join(args.seq_tfrecords_file_path, 'tran.tfrecords')
    with tf.device('/cpu:0'):
        realBatchSize = args.batch_size
        if args.dataset_type == 'multiple':
            dataset1 = tf.data.TFRecordDataset(seq_tfrecords_f)
            dataset1 = dataset1.map(parse_function)
            realBatchSize = realBatchSize//2
            dataset1 = dataset1.shuffle(buffer_size=args.buffer_size)            
            dataset1 = dataset1.apply(tf.contrib.data.batch_and_drop_remainder(realBatchSize))
            iterator1 = dataset1.make_initializable_iterator()
            next_element1 = iterator1.get_next()
            
        dataset = tf.data.TFRecordDataset(id_tfrecords_f)
        dataset = dataset.map(distortion_parse_function)
        dataset = dataset.shuffle(buffer_size=args.buffer_size)
        #dataset = dataset.batch(realBatchSize)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(realBatchSize))
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

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
    #p = int(512.0/args.batch_size)
    #lr_steps = [p*val for val in args.lr_steps]
    #print(lr_steps)
    #logging.info(lr_steps)    
    if len(args.lr_steps)==0:
        lr_steps = [40000, 60000, 80000]
        #p = int(512.0/args.batch_size)
        #lr_steps = [p*val for val in args.lr_steps]        
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('learning rate steps: ', lr_steps)
    logging.info(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001],name='lr_schedule')
    grad_factor = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.1, 0.3, 0.5, 0.8], name='grad_schedule')
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
            net = get_resnet(images_s[i], args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
            
            # 3.4 get arcface loss
            logit = arcface_loss(embedding=net.outputs, labels=labels_s[i], w_init=w_init_method, out_num=args.id_num_output)
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            
            if args.dataset_type == 'multiple':
                # 3.4.a split logits and labels into identity dataset and sequence dataset
                idLogits, seqLogits = tf.split(logit,2,0)
                idLabels, seqLabels = tf.split(labels_s[i],2,0)
            else:
                idLogits = logit
                idLabels = labels_s[i]
    
            # define the cross entropy added LSR parts  chief loss
            identity_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=idLogits, labels=idLabels))
            if (args.dataset_type == 'multiple') and (args.lsr):
                sequence_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.nn.softmax(seqLogits),1e-30, 1))) # warning
                chief_loss = identity_loss + sequence_loss*args.sequence_loss_factor
            else:
                chief_loss = identity_loss
                sequence_loss = None
            
            # 3.3.a auxiliary loss
            if args.aux_loss_type == 'center':
                if args.dataset_type == 'single':
                    logits_center_loss, _ = center_loss(net.outputs, labels, args.center_loss_alfa, args.id_num_output)
                else:
                    logits_center_loss, _ = center_loss(net.outputs, labels, args.center_loss_alfa, args.id_num_output+args.seq_num_output)
                auxiliary_loss = logits_center_loss    
            elif args.aux_loss_type == 'dsa':
                if args.dataset_type == 'single':
                    feature_dsa_loss, _ = single_dsa_loss(net.outputs, labels, args.center_loss_alfa, args.id_num_output, args.dsa_param, args.batch_size)
                else:
                    feature_dsa_loss, _ = multiple_dsa_loss(net.outputs, labels, args.center_loss_alfa, args.id_num_output, args.seq_num_output, args.dsa_param, args.batch_size)
                auxiliary_loss = feature_dsa_loss    
            else:
                auxiliary_loss = None
            
            # define weight deacy losses
            wd_loss = 0
            for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
                #if (args.pretrained_model) and not (('E_DenseLayer' in weights.name) or ('E_BN2' in weights.name)):
                #    continue
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
            for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
                #if (args.pretrained_model) and not (('E_DenseLayer' in weights.name) or ('E_BN2' in weights.name)):
                #    continue
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
            for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
                #if (args.pretrained_model) and not (('E_DenseLayer' in weights.name) or ('E_BN2' in weights.name)):
                #    continue
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
            for gamma in tl.layers.get_variables_with_name('gamma', True, True):
                #if (args.pretrained_model) and not (('E_DenseLayer' in weights.name) or ('E_BN2' in weights.name)):
                #    continue
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
            for alphas in tl.layers.get_variables_with_name('alphas', True, True):
                #if (args.pretrained_model) and not (('E_DenseLayer' in weights.name) or ('E_BN2' in weights.name)):
                #    continue
                wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
                
            #total_loss
            if args.aux_loss_type != None:
                total_loss = chief_loss + auxiliary_loss * args.auxiliary_loss_factor + wd_loss*args.norm_loss_factor
            else:
                total_loss = chief_loss + wd_loss*args.norm_loss_factor

            loss_dict[('identity_loss_%s_%d' % ('gpu', i))] = identity_loss
            loss_keys.append(('identity_loss_%s_%d' % ('gpu', i)))
            if (sequence_loss != None):
                loss_dict[('sequence_loss_%s_%d' % ('gpu', i))] = sequence_loss
                loss_keys.append(('sequence_loss_%s_%d' % ('gpu', i)))
            loss_dict[('chief_loss_%s_%d' % ('gpu', i))] = chief_loss
            loss_keys.append(('chief_loss_%s_%d' % ('gpu', i)))
            if (auxiliary_loss != None):            
                loss_dict[('auxiliary_loss_%s_%d' % ('gpu', i))] = auxiliary_loss
                loss_keys.append(('auxiliary_loss_%s_%d' % ('gpu', i)))
            loss_dict[('wd_loss_%s_%d' % ('gpu', i))] = wd_loss
            loss_keys.append(('wd_loss_%s_%d' % ('gpu', i)))
            loss_dict[('total_loss_%s_%d' % ('gpu', i))] = total_loss
            loss_keys.append(('total_loss_%s_%d' % ('gpu', i)))           
            
            cur_trainable_vals = tf.trainable_variables()
            real_trainable_vals = []
            variable_map = {}
            #if args.pretrained_model:        
            #    cur_trainable_names = [v.name.split(':')[0] for v in cur_trainable_vals] # val list
            #    pretrained_vals = tf.train.list_variables(args.pretrained_model) # val tuples list (name, shape)
            #    pretrained_names = [v[0] for v in pretrained_vals]
            #    for name in cur_trainable_names:
            #        if (name in pretrained_names) and not('arcface_loss' in name):
            #            variable_map[name] = name   # vals to be initialized
            #        if ('E_DenseLayer' in name) or ('E_BN2' in name) or ('arcface_loss' in name) or (name not in pretrained_names):
            #            real_trainable_vals.append(name) # stop gradients
            #    needed_trainable_vals = [v for v in cur_trainable_vals if v.name.split(':')[0] in real_trainable_vals]
            
            grads = opt.compute_gradients(total_loss, var_list=cur_trainable_vals) #warning: gradients stopping                
            tower_grads.append(grads)
            if i == 0:
                test_net = get_resnet(images_test, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
                embedding_tensor = test_net.outputs
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                pred = tf.nn.softmax(idLogits)
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), idLabels), dtype=tf.float32))

    grads = average_gradients(tower_grads)
    #modify mult-lr    
    grads_and_vars_mult = []
    for grad, var in grads:
        #if "embedding_weights" not in var.op.name:
        #if (('E_DenseLayer' in var.op.name) or ('E_BN2' in var.op.name)) and (grad != None):
        #    grad *= grad_factor
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
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summary_op = tf.summary.merge(summaries)

    # Create a saver.
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # init all variables
    #sess.run(tf.global_variables_initializer())    #if restore by saver
    #sess.run(tf.local_variables_initializer())
    if args.pretrained_model:
        logging.info('Restoring pretrained model: %s' % args.pretrained_model)
        # 3.12b pretrained model saver
        #saver.restore(sess, args.pretrained_model)
        tf.train.init_from_checkpoint(os.path.dirname(args.pretrained_model), variable_map)
        #xm = {'resnet_v1_50/conv1/W_conv2d':'resnet_v1_50/conv1/W_conv2d'}
        #tf.train.init_from_checkpoint(args.pretrained_model, xm)
    sess.run(tf.global_variables_initializer())    # if initializing by init_from_checkpoint.
    sess.run(tf.local_variables_initializer())

    sess.graph.finalize()
    # begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    
    count = 0
    total_accuracy = {}
    for i in range(args.epoch):
        sess.run(iterator.initializer)
        if args.dataset_type == 'multiple':
            sess.run(iterator1.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                if args.dataset_type == 'multiple':
                    images_train1, labels_train1 = sess.run(next_element1)                    
                    images_train_s = np.split(images_train, args.num_gpus, 0)
                    labels_train_s = np.split(labels_train, args.num_gpus, 0)
                    images_train1_s = np.split(images_train1, args.num_gpus, 0)
                    labels_train1_s = np.split(labels_train1, args.num_gpus, 0)
                    train_data = None
                    label_data = None
                    for i in range(args.num_gpus):
                        data_block = np.concatenate((images_train_s[i], images_train1_s[i]), axis=0)
                        label_block = np.concatenate((labels_train_s[i], labels_train1_s[i]), axis=0)
                        if train_data is None and label_data is None:
                            train_data = data_block
                            label_data = label_block
                        else:
                            train_data = np.concatenate((train_data, data_block), axis=0)
                            label_data = np.concatenate((label_data, label_block), axis=0)
                else:
                    train_data = images_train
                    label_data = labels_train
                feed_dict = {images: train_data, labels: label_data, dropout_rate: 0.4}
                start = time.time()
                
                rsltList = [None, None, None, None, None, None, None, None, None, None] # trainOpVal, total_loss_val1, chief_loss_val1, identity_loss_val1, wd_loss_val1, total_loss_val2, chief_loss_val2, identity_loss_val2, wd_loss_val2, acc_val
                opList = [train_op, loss_dict[('total_loss_%s_%d' % ('gpu', 0))], loss_dict[('chief_loss_%s_%d' % ('gpu', 0))], loss_dict[('identity_loss_%s_%d' % ('gpu', 0))], loss_dict[('wd_loss_%s_%d' % ('gpu', 0))], loss_dict[('total_loss_%s_%d' % ('gpu', 1))], loss_dict[('chief_loss_%s_%d' % ('gpu', 1))], loss_dict[('identity_loss_%s_%d' % ('gpu', 1))], loss_dict[('wd_loss_%s_%d' % ('gpu', 1))], acc]
                if args.aux_loss_type != None:
                    rsltList.append(None)# auxiliary_loss_val
                    rsltList.append(None)
                    opList.append(loss_dict[('auxiliary_loss_%s_%d' % ('gpu', 0))])
                    opList.append(loss_dict[('auxiliary_loss_%s_%d' % ('gpu', 1))])
                if args.lsr:
                    rsltList.append(None) # sequence_loss_val
                    rsltList.append(None) # sequence_loss_val
                    opList.append(loss_dict[('sequence_loss_%s_%d' % ('gpu', 0))])
                    opList.append(loss_dict[('sequence_loss_%s_%d' % ('gpu', 1))])
                
                rsltList = sess.run(opList, feed_dict=feed_dict)
                
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                
                if len(rsltList) < 14:
                    rsltList = rsltList + [rsltList[-2], rsltList[-1]]*((14-len(rsltList)//2))
                
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss: [%.2f, %.2f], chief loss: [%.2f, %.2f], identity loss: [%.2f, %.2f], sequence loss: [%.2f, %.2f], auxiliary loss: [%.2f, %.2f], weight deacy loss: [%.2f, %.2f], training accuracy is %.6f, time %.3f samples/sec' % (i, count, rsltList[1], rsltList[5], rsltList[2], rsltList[6], rsltList[3], rsltList[7], rsltList[-2], rsltList[-1], rsltList[10], rsltList[11], rsltList[4], rsltList[8], rsltList[9], pre_sec))
                    logging.info('epoch %d, total_step %d, total loss: [%.2f, %.2f], chief loss: [%.2f, %.2f], identity loss: [%.2f, %.2f], sequence loss: [%.2f, %.2f], auxiliary loss: [%.2f, %.2f], weight deacy loss: [%.2f, %.2f], training accuracy is %.6f, time %.3f samples/sec' % (i, count, rsltList[1], rsltList[5], rsltList[2], rsltList[6], rsltList[3], rsltList[7], rsltList[-2], rsltList[-1], rsltList[10], rsltList[11], rsltList[4], rsltList[8], rsltList[9], pre_sec))      
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
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess, embedding_tensor=embedding_tensor, batch_size=args.batch_size//args.num_gpus, feed_dict=feed_dict_test, input_placeholder=images_test)
                    if len(results) > 0:
                        logging.info("lfw test accuracy is: %.5f" % (results[0]))
                        total_accuracy[str(count)] = results[0]
                        log_file.write('########'*10+'\n')
                        log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                        log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                        log_file.flush()
                        if max(results) > 0.995:
                            print('best accuracy is %.5f' % max(results))
                            filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                            filename = os.path.join(args.ckpt_path, filename)
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
