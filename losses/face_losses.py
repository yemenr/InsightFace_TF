import tensorflow as tf
import math
import random

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    with tf.device('/cpu:0'):
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])#all labels
    centers_batch = tf.gather(centers, label)#get centers batch
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)#更新指定位置centers 指定位置作差
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))#center loss 更新完centers再做新一轮计算
    return loss, centers

def single_dsa_loss(features, label, alfa, id_num, dsa_param, batch_size):    
    dsa_lambda, dsa_alpha, dsa_beta, dsa_p = dsa_param    
    idFeatures = features
    idLabels = label
    nrof_features = features.get_shape()[1]
    
    with tf.device('/cpu:0'):
        centers = tf.get_variable('centers', [id_num, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])#all labels
    centers_batch = tf.gather(centers, label)#get centers batch
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)#更新指定位置centers 指定位置作差   
    
    id_inter_cnt = int(id_num*dsa_p)        
    id_label_selected = [x for x in range(id_num)]
    random.shuffle(id_label_selected)
    id_label_selected = id_label_selected[:id_inter_cnt]
    
    id_selected_centers = tf.gather(centers, id_label_selected)
    
    id_label_selected = id_label_selected*(batch_size)
    id_selected_centers = tf.tile(id_selected_centers, [batch_size,1])
    
    idFeaturesSelected = tf.reshape(tf.tile(tf.reshape(idFeatures,[-1,1,nrof_features]),[1,id_inter_cnt,1]),[-1,nrof_features])
    idLabelsExt = tf.reshape(tf.tile(tf.reshape(idLabels,[-1,1]),[1,id_inter_cnt]),[-1])
    id_selected = tf.not_equal(tf.convert_to_tensor(id_label_selected, dtype=tf.int64),idLabelsExt)
    
    with tf.control_dependencies([centers]):
        intra_center_diff = tf.reduce_mean(tf.square(features - centers_batch),1,keep_dims=True)
        center_loss_part = tf.reduce_mean(intra_center_diff)#center loss 更新完centers再做新一轮计算
        
        idSelectedDiff = intra_center_diff        
        idSelectedDiff = tf.reshape(tf.tile(tf.reshape(idSelectedDiff,[-1,1,1]),[1,id_inter_cnt,1]),[-1,1])
                
        #identity dataset inter class loss part
        idSelectedInterDiff = tf.reduce_mean(tf.square(idFeaturesSelected-id_selected_centers),1,keep_dims=True)        
        idSelectedInterZeros = tf.zeros_like(idSelectedInterDiff)
        idSelectedInterDiff = dsa_alpha*idSelectedDiff-idSelectedInterDiff+dsa_beta
        idSelectedInterDiff = tf.where(idSelectedInterDiff>0,idSelectedInterDiff,idSelectedInterZeros)
        idSelectedInterDiff = tf.reduce_mean(tf.where(id_selected,idSelectedInterDiff,idSelectedInterZeros))
        inter_loss_part = idSelectedInterDiff
        
        loss = dsa_lambda*center_loss_part + (1-dsa_lambda)*inter_loss_part
    return loss, centers   

def multiple_dsa_loss(features, label, alfa, id_num, seq_num, dsa_param, batch_size):
    idFeatures, seqFeatures = tf.split(features,2,0)
    idLabels, seqLabels = tf.split(label,2,0)
    dsa_lambda, dsa_alpha, dsa_beta, dsa_p = dsa_param
    
    #batch_size = features.get_shape()[0]
    nrof_features = features.get_shape()[1]
    
    centers = tf.get_variable('centers', [id_num+seq_num, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])#all labels
    centers_batch = tf.gather(centers, label)#get centers batch
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)#更新指定位置centers 指定位置作差    
    
    id_inter_cnt = int((id_num+seq_num)*dsa_p)
    seq_inter_cnt = int(id_num*dsa_p)
        
    id_label_selected = [x for x in range(id_num+seq_num)]
    random.shuffle(id_label_selected)
    id_label_selected = id_label_selected[:id_inter_cnt]
    seq_label_selected = [x for x in range(id_num)]
    random.shuffle(seq_label_selected)
    seq_label_selected = seq_label_selected[:seq_inter_cnt]
    
    id_selected_centers = tf.gather(centers, id_label_selected)
    seq_selected_centers = tf.gather(centers, seq_label_selected)
    
    id_label_selected = id_label_selected*(batch_size//2)
    seq_label_selected = seq_label_selected*(batch_size//2)
    id_selected_centers = tf.tile(id_selected_centers, [batch_size//2,1])
    seq_selected_centers = tf.tile(seq_selected_centers, [batch_size//2,1])
    
    idFeaturesSelected = tf.reshape(tf.tile(tf.reshape(idFeatures,[-1,1,nrof_features]),[1,id_inter_cnt,1]),[-1,nrof_features])
    idLabelsExt = tf.reshape(tf.tile(tf.reshape(idLabels,[-1,1]),[1,id_inter_cnt]),[-1])
    seqFeaturesSelected = tf.reshape(tf.tile(tf.reshape(seqFeatures,[-1,1,nrof_features]),[1,seq_inter_cnt,1]),[-1,nrof_features])
    seqLabelsExt = tf.reshape(tf.tile(tf.reshape(seqLabels,[-1,1]),[1,seq_inter_cnt]),[-1])
    id_selected = tf.not_equal(tf.convert_to_tensor(id_label_selected, dtype=tf.int64),idLabelsExt)
    seq_selected = tf.not_equal(tf.convert_to_tensor(seq_label_selected, dtype=tf.int64),seqLabelsExt)
    
    with tf.control_dependencies([centers]):
        intra_center_diff = tf.reduce_mean(tf.square(features - centers_batch),1,keep_dims=True)
        center_loss_part = tf.reduce_mean(intra_center_diff)#center loss 更新完centers再做新一轮计算
        
        idSelectedDiff, seqSelectedDiff = tf.split(intra_center_diff,2,0)        
        idSelectedDiff = tf.reshape(tf.tile(tf.reshape(idSelectedDiff,[-1,1,1]),[1,id_inter_cnt,1]),[-1,1])
        seqSelectedDiff = tf.reshape(tf.tile(tf.reshape(seqSelectedDiff,[-1,1,1]),[1,seq_inter_cnt,1]),[-1,1])
                
        #identity dataset inter class loss part
        idSelectedInterDiff = tf.reduce_mean(tf.square(idFeaturesSelected-id_selected_centers),1,keep_dims=True)        
        idSelectedInterZeros = tf.zeros_like(idSelectedInterDiff)
        idSelectedInterDiff = dsa_alpha*idSelectedDiff-idSelectedInterDiff+dsa_beta
        idSelectedInterDiff = tf.where(idSelectedInterDiff>0,idSelectedInterDiff,idSelectedInterZeros)
        idSelectedInterDiff = tf.reduce_mean(tf.where(id_selected,idSelectedInterDiff,idSelectedInterZeros))
        #sequence dataset inter class loss part
        seqSelectedInterDiff = tf.reduce_mean(tf.square(seqFeaturesSelected-seq_selected_centers),1,keep_dims=True)        
        seqSelectedInterZeros = tf.zeros_like(seqSelectedInterDiff)
        seqSelectedInterDiff = dsa_alpha*seqSelectedDiff-seqSelectedInterDiff+dsa_beta
        seqSelectedInterDiff = tf.where(seqSelectedInterDiff>0,seqSelectedInterDiff,seqSelectedInterZeros)
        seqSelectedInterDiff = tf.reduce_mean(tf.where(seq_selected,seqSelectedInterDiff,seqSelectedInterZeros))
        inter_loss_part = idSelectedInterDiff+seqSelectedInterDiff
        
        loss = dsa_lambda*center_loss_part + (1-dsa_lambda)*inter_loss_part
    return loss, centers    
    
def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     0<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)   #tricks : additive margin instead
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


def combine_loss_val(embedding, labels, w_init, out_num, margin_a, margin_m, margin_b, s):
    '''
    This code is contributed by RogerLo. Thanks for you contribution.

    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                              initializer=w_init, dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
    cos_t = tf.matmul(embedding_unit, weights_unit)
    ordinal = tf.constant(list(range(0, embedding.get_shape().as_list()[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, labels], axis=1)
    zy = cos_t * s
    sel_cos_t = tf.gather_nd(zy, ordinal_y)
    if margin_a != 1.0 or margin_m != 0.0 or margin_b != 0.0:
        if margin_a == 1.0 and margin_m == 0.0:
            s_m = s * margin_b
            new_zy = sel_cos_t - s_m
        else:
            cos_value = sel_cos_t / s
            t = tf.acos(cos_value)
            if margin_a != 1.0:
                t = t * margin_a
            if margin_m > 0.0:
                t = t + margin_m
            body = tf.cos(t)
            if margin_b > 0.0:
                body = body - margin_b
            new_zy = body * s
    updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), zy.get_shape()))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
    predict_cls = tf.argmax(updated_logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    predict_cls_s = tf.argmax(zy, 1)
    accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls_s, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    return zy, loss, accuracy, accuracy_s, predict_cls_s