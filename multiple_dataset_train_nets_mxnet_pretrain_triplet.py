import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import parse_function, distortion_parse_function
import os
# from nets.L_Resnet_E_IR import get_resnet
# from nets.L_Resnet_E_IR_GBN import get_resnet
from nets.mxnet_pretrain import get_resnet
from losses.face_losses import arcface_loss, center_loss, single_dsa_loss, multiple_dsa_loss, single_git_loss, multiple_git_loss, triplet_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test
import logging
import pdb
import numpy as np

args = None

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=50,type=int, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, type=int, help='epoch to train the network')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size to train network')
    parser.add_argument('--lr_steps', type=str, default='', help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=8e-4, type=float, help='learning alg momentum')
    #parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'survellance'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['surveillance','lfw'], help='evluation datasets')
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
    #parser.add_argument('--dsa_param', default=[0.5, 2, 1, 0.3], help='[dsa_lambda, dsa_alpha, dsa_beta, dsa_p]')
    parser.add_argument('--dsa_param', default=[0.5, 0.1, 1, 1], help='[dsa_lambda, dsa_alpha, dsa_beta, dsa_p]')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', type=int, default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', default=None, help='pretrained model')
    parser.add_argument('--devices', default='0', help='the ids of gpu devices')
    parser.add_argument('--log_file_name', default='train_out.log', help='the ids of gpu devices')
    parser.add_argument('--dataset_type', default='multiple', help='single dataset or multiple dataset')
    parser.add_argument('--lsr', action='store_true', help='add LSR item')
    parser.add_argument('--aux_loss_type', default=None, help='None | center | dsa loss | git loss')
    parser.add_argument('--weight_file', default=None, help='mxnet r100 weight file')
    parser.add_argument('--triplet_loss', action='store_true', help='use triplet loss')
    parser.add_argument('--triplet_alpha', type=float, help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--num_classes_per_batch', default=18, type=int, help='epoch to train the network')
    parser.add_argument('--num_images_per_class', default=50, type=int, help='epoch to train the network')
    args = parser.parse_args()
    return args

def generator():
    global args
    while True:
        # Sample the labels that will compose the batch
        labels = np.random.choice(range(args.id_num_output),
                                  args.num_classes_per_batch,
                                  replace=False)
        for label in labels:
            for _ in range(args.num_images_per_class):
                yield label    

def cal_and_select_triplets(embeddings, args, imagesTrain, labelsTrain):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    tripletsLabel = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    nrof_images = int(args.num_images_per_class)
    for i in xrange(args.num_classes_per_batch):        
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((imagesTrain[a_idx], imagesTrain[p_idx], imagesTrain[n_idx]))
                    tripletsLabel.append((labelsTrain[a_idx], labelsTrain[p_idx], labelsTrain[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    triplets = np.concatenate(triplets, axis=0)
    tripletsLabel = np.concatenate(tripletsLabel, axis=0)
    print("candidate triplets count: %d, validate triplets count: %d" % (num_trips, len(triplets)//3))
    return triplets, tripletsLabel
                
def select_triplets_images(sess, imagesPlaceholder, labelsPlaceholder, dropoutRatePlaceholder, next_element, realBatchSize, net):
    tripletsImages = np.zeros((realBatchSize, 112, 112, 3))
    embeddingSize = 512
    selectedCnt = 0
    while(selectedCnt < realBatchSize):
        # produce enough id data batch
        images_train, labels_train = sess.run(next_element)
        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = len(images_train)
        emb_array = np.zeros((nrof_examples, embeddingSize))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {imagesPlaceholder: images_train[i*args.batch_size:i*args.batch_size+batch_size,...], dropoutRatePlaceholder: 0.4}
            emb = sess.run([net], feed_dict=feed_dict)
            emb_array[i*args.batch_size:i*args.batch_size+batch_size,:] = emb
        print('%.3f' % (time.time()-start_time))
        
        ## calculate and select triplets
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = cal_and_select_triplets(emb_array, args, images_train, labels_train)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (nrof_random_negs, nrof_triplets, selection_time))
            
        ## concatenate triplets
        if (selectedCnt < realBatchSize):
            selCnt = min(len(triplets), realBatchSize-selectedCnt)
            tripletsImages[slectedCnt:selCnt+Cnt,...] = triplets
            selectedCnt += selCnt
                
    return tripletsImages, tripletsLabels
    
def train_process(args, sess, imagesPlaceholder, labelsPlaceholder, dropoutRatePlaceholder, iterator, next_element, net, anchor,
                  positive, negative, embedding_tensor, realBatchSize, summary_op, saver, log_file, embedding_tensor, 
                  next_element1=None, iterator1=None):
    # 4 begin iteration
    count = 0

    for i in range(args.epoch):
        sess.run(iterator.initializer)
        if args.dataset_type == 'multiple':
            sess.run(iterator1.initializer)
        while True:
            try:
                # produce images of triplets
                images_train, labels_train = select_triplets_images(sess, imagesPlaceholder, labelsPlaceholder, dropoutRatePlaceholder, next_element, realBatchSize, net)
                
                # get id and seq data
                if args.dataset_type == 'multiple':
                    images_train1, labels_train1 = sess.run(next_element1)
                    train_data = np.concatenate((images_train, images_train1), axis=0)
                    label_data = np.concatenate((labels_train, labels_train1), axis=0)
                else:
                    train_data = images_train
                    label_data = labels_train
                feed_dict = {images: train_data, labels: label_data, dropout_rate: 0.4}
                start = time.time()
                
                rsltList = [None, None, None, None] # trainOpVal, total_loss_val, chief_loss_val, wd_loss_val
                opList = [train_op, total_loss, chief_loss, wd_loss]
                if args.aux_loss_type != None:
                    rsltList.append(None) # auxiliary_loss_val
                    opList.append(auxiliary_loss)
                
                rsltList = sess.run(opList, feed_dict=feed_dict,
                                    options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                
                #change_map = {}
                #for name in list(var_map.keys()):
                #    x = sess.run(var_map[name][0])
                #    if not np.array_equal(x,var_map[name][1]):
                #        change_map[name] = [var_map[name][1]]
                #        change_map[name].append(x)

                #print(test_0)
                #print(test_1)
                #pdb.set_trace()
                if len(rsltList) < 5:
                    rsltList = rsltList + [rsltList[-1]]*(5-len(rsltList))
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , chief loss is %.2f, auxiliary loss is %.2f, weight deacy loss is %.2f, time %.3f samples/sec' % (i, count, rsltList[1], rsltList[2], rsltList[-1], rsltList[3], pre_sec))
                    logging.info('epoch %d, total_step %d, total loss is %.2f , chief loss is %.2f, auxiliary loss is %.2f, weight deacy loss is %.2f, time %.3f samples/sec' % (i, count, rsltList[1], rsltList[2], rsltList[-1], rsltList[3], pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: train_data, labels: label_data, dropout_rate: 0.4}
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
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                             input_placeholder=images)
                    if len(results) > 0:
                        print('test accuracy is: ', str(results[0]))
                        logging.info('test accuracy is: %s' % str(results[0]))
                        total_accuracy[str(count)] = results[0]
                        log_file.write('########'*10+'\n')
                        log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                        log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                        log_file.flush()
                        if max(results) >= 0.816:
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
                
if __name__ == '__main__':
    global args
    args = get_parser()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename = args.log_file_name,level=logging.INFO, format = log_format)
        
    # 1. define global parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    images = tf.placeholder(name='input', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=[], name='dropout_rate')
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    idTfRecNames = [os.path.join(args.id_tfrecords_file_path, 'tran.tfrecords'+str(k)) for k in range(1,args.id_num_output+1)]    
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
        
        per_class_datasets = [tf.data.TFRecordDataset(f).repeat(None).map(distortion_parse_function) for f in idTfRecNames]
        choice_dataset = tf.data.Dataset.from_generator(generator, tf.int64)
        dataset = tf.contrib.data.choose_from_datasets(per_class_datasets, choice_dataset)
        candTripletsBatchSize = args.num_classes_per_batch * args.num_images_per_class
        dataset = dataset.batch(candTripletsBatchSize)
        dataset = dataset.prefetch(None)
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
    _, net = get_resnet(images, w_init=w_init_method, trainable=True, keep_rate=dropout_rate, weight_file=args.weight_file)
    # test net  because of batch normal layer
    test_net, _ = get_resnet(images, w_init=w_init_method, trainable=False, reuse=True, keep_rate=dropout_rate)
    embedding_tensor = test_net
    
    # Split embeddings into anchor, positive and negative and calculate triplet loss
    anchor, positive, negative = tf.unstack(tf.reshape(net, [-1,3,512]), 3, 1)
    
    # 3.2 get arcface loss
    tripletLoss = triplet_loss(anchor, positive, negative, args.triplet_alpha)
    chief_loss = tripletLoss        
        
    # 3.3.a auxiliary loss
    if args.aux_loss_type == 'center':
        if args.dataset_type == 'single':
            logits_center_loss, _ = center_loss(net, labels, args.center_loss_alfa, args.id_num_output)
        else:
            logits_center_loss, _ = center_loss(net, labels, args.center_loss_alfa, args.id_num_output+args.seq_num_output)
        auxiliary_loss = logits_center_loss    
    elif args.aux_loss_type == 'dsa':
        if args.dataset_type == 'single':
            feature_dsa_loss, _ = single_dsa_loss(net, labels, args.center_loss_alfa, args.id_num_output, args.dsa_param, args.batch_size)
        else:
            feature_dsa_loss, _ = multiple_dsa_loss(net, labels, args.center_loss_alfa, args.id_num_output, args.seq_num_output, args.dsa_param, args.batch_size)
        auxiliary_loss = feature_dsa_loss  
    elif args.aux_loss_type == 'git':
        if args.dataset_type == 'single':
            feature_git_loss, _ = single_git_loss(net, labels, args.center_loss_alfa, args.id_num_output, args.dsa_param, args.batch_size)
        else:
            feature_git_loss, _ = multiple_git_loss(net, labels, args.center_loss_alfa, args.id_num_output, args.seq_num_output, args.dsa_param, args.batch_size)
        auxiliary_loss = feature_git_loss        
    else:
        auxiliary_loss = None
        
    # 3.4 define weight deacy losses
    # for var in tf.trainable_variables():
    #     print(var.name)
    # print('##########'*30)
    wd_loss = 0
    for weights in tl.layers.get_variables_with_name('weights', True, True): # all weight   
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for W in tl.layers.get_variables_with_name('kernel', True, True): # dense layer
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
    for gamma in tl.layers.get_variables_with_name('gamma', True, True): # prelu
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)

    # 3.5 total losses
    if args.aux_loss_type != None:
        total_loss = chief_loss + auxiliary_loss * args.auxiliary_loss_factor + wd_loss*args.norm_loss_factor
    else:
        total_loss = chief_loss + wd_loss*args.norm_loss_factor
    
    # 3.6 define the learning rate schedule
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
    #print(lr_steps)
    
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    
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
    #        if ('fc1' in name) or ('arcface_loss' in name) or (name not in pretrained_names):
    #            real_trainable_vals.append(name) # stop gradients
    #    needed_trainable_vals = [v for v in cur_trainable_vals if v.name.split(':')[0] in real_trainable_vals]
    
    grad_factor = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.0000000, 0.00000001, 0.00000001, 0.000000001], name='grad_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
    # 3.8 get train op
    #with tf.device('/cpu:0'):
    grads = opt.compute_gradients(total_loss, var_list=cur_trainable_vals) #warning: gradients stopping
    #modify mult-lr if needed
    grads_and_vars_mult = []
    for grad, var in grads:
        if "embedding_weights" not in var.op.name:
        #if (('E_DenseLayer' in var.op.name) or ('E_BN2' in var.op.name)) and (grad != None):
            grad *= grad_factor
        grads_and_vars_mult.append((grad, var))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads_and_vars_mult, global_step=global_step)
        
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
    if (auxiliary_loss != None):
        summaries.append(tf.summary.scalar('auxiliary_loss', auxiliary_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    
    #tw0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut_conv/W_conv2d")[0]
    #tw1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v1_50/E_DenseLayer/W")[0]
    
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())    #if restore by saver
    sess.run(tf.local_variables_initializer())
    if args.pretrained_model:
        logging.info('Restoring pretrained model: %s' % args.pretrained_model)
        # 3.12b pretrained model saver
        saver.restore(sess, args.pretrained_model)
   
    #var_map = {}
    #for name in list(variable_map.keys()):
    #    var_var = [v for v in tf.trainable_variables() if name == v.name.split(':')[0]][0]
    #    var_map[name] = [var_var]
    #    var_value = sess.run(var_var)
    #    var_map[name].append(var_value)

    #test_count = 0
    #for i in range(0):
    #    test_count += 1
    #    feed_dict_test ={dropout_rate: 1.0}
    #    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
    #    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=test_count, sess=sess,
    #                         embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
    #                         input_placeholder=images)

    # 4 begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    train_process(args, sess, images, labels, dropout_rate, iterator, next_element, net, anchor,
                  positive, negative, embedding_tensor, realBatchSize, summary_op, saver, log_file, embedding_tensor,
                  next_element1, iterator1)
    log_file.close()
    log_file.write('\n')