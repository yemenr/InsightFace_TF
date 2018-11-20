import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers.python.layers import utils
import collections
from .tl_layers_modify import ElementwiseLayer, BatchNormLayer, Conv2d, PReluLayer, DenseLayer
import numpy as np

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return tl.layers.MaxPool2d(inputs, [1, 1], strides=(factor, factor), name=scope)


def conv2d_same(inputs, num_outputs, kernel_size, strides, rate=1, w_init=None, scope=None, trainable=None):
    '''
    Reference slim resnet
    :param inputs:
    :param num_outputs:
    :param kernel_size:
    :param strides:
    :param rate:
    :param scope:
    :return:
    '''
    if strides == 1:
        if rate == 1:
            nets = Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                   strides=(strides, strides), W_init=w_init, act=None, padding='SAME', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        return nets
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tl.layers.PadLayer(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], name='padding_%s' % scope)
        if rate == 1:
            nets = Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), W_init=w_init, act=None, padding='VALID', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                              rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        return nets


def bottleneck_IR(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv', use_cudnn_on_gpu=True)
            shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=True, trainable=trainable, name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = BatchNormLayer(inputs, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn1')
        residual = Conv2d(residual, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None, b_init=None,
                                    W_init=w_init, name='conv1', use_cudnn_on_gpu=True)
        residual = BatchNormLayer(residual, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn2')
        # bottleneck prelu
        residual = PReluLayer(residual)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2', trainable=trainable)
        output = ElementwiseLayer(layer=[shortcut, residual],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=None)
        return output


def bottleneck_IR_SE(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv', use_cudnn_on_gpu=True)
            shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=True, trainable=trainable, name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = BatchNormLayer(inputs, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn1')
        residual = Conv2d(residual, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None, b_init=None,
                                    W_init=w_init, name='conv1', use_cudnn_on_gpu=True)
        residual = BatchNormLayer(residual, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn2')
        # bottleneck prelu
        residual = PReluLayer(residual)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2', trainable=trainable)
        # squeeze
        squeeze = tl.layers.InputLayer(tf.reduce_mean(residual.outputs, axis=[1, 2]), name='squeeze_layer')
        # excitation
        excitation1 = DenseLayer(squeeze, n_units=int(depth/16.0), act=tf.nn.relu,
                                           W_init=w_init, name='excitation_1')
        # excitation1 = tl.layers.PReluLayer(excitation1, name='excitation_prelu')
        excitation2 = DenseLayer(excitation1, n_units=depth, act=tf.nn.sigmoid,
                                           W_init=w_init, name='excitation_2')
        # scale
        scale = tl.layers.ReshapeLayer(excitation2, shape=[tf.shape(excitation2.outputs)[0], 1, 1, depth], name='excitation_reshape')

        residual_se = ElementwiseLayer(layer=[residual, scale],
                                       combine_fn=tf.multiply,
                                       name='scale_layer',
                                       act=None)

        output = ElementwiseLayer(layer=[shortcut, residual_se],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=tf.nn.relu)
        return output

def stn_process_tl(inputs, trainable):
    #spatial transformer
    #use one channel value
    #input_for_theta = tf.expand_dims(inputs[:,:,:,-1],axis=3)
    stn_inputs = tl.layers.InputLayer(inputs, name='stn_input_layer')
    #conv layer1: (5,5) kernels * 24, stride=1, no padding    
    conv1_loc = Conv2d(stn_inputs, n_filter=24, filter_size=(5, 5), strides=(1, 1),
                       act=None, padding='VALID', W_init=tf.contrib.layers.xavier_initializer(), 
                       b_init=None, name='Conv2d_1_5x5', use_cudnn_on_gpu=True)#?,124,124,24 weights initializer modified    
    #batch normal layer1                                 
    bn1_loc = BatchNormLayer(conv1_loc, act=tf.identity, is_train=True, name='bn1_loc', trainable=trainable)
    #pooling layer1(contains prelu): max pooling kernel_size=2, stride=2
    pool1_loc = tl.layers.MaxPool2d(bn1_loc, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='MaxPool_1_2x2')#?,62,62,24
    #prelu
    prelu1_loc = PReluLayer(pool1_loc, name='prelu1_loc')
    
    #conv layer2: (3,3) kernels * 48, stride=1, padding=0
    conv2_loc = Conv2d(prelu1_loc, n_filter=48, filter_size=(3, 3), strides=(1, 1),
                                 act=None, padding='VALID', W_init=tf.contrib.layers.xavier_initializer(), 
                                 b_init=None, name='Conv2d_2_3x3', use_cudnn_on_gpu=True)#?,60,60,48
    #batch normal layer1                                 
    bn2_loc = BatchNormLayer(conv2_loc, act=tf.identity, is_train=True, name='bn2_loc', trainable=trainable)
    #pooling layer2(contains prelu): max pooling kernel_size=2, stride=2
    pool2_loc = tl.layers.MaxPool2d(bn2_loc, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='MaxPool_2_2x2')#?,30,30,48
    #prelu
    prelu2_loc = PReluLayer(pool2_loc, name='prelu2_loc')    
    
    #conv layer3: (3,3) kernels * 96, stride=1, padding=0
    conv3_loc = Conv2d(prelu2_loc, n_filter=96, filter_size=(3, 3), strides=(1, 1),
                       act=None, padding='VALID', W_init=tf.contrib.layers.xavier_initializer(), 
                       b_init=None, name='Conv2d_3_3x3', use_cudnn_on_gpu=True)#?,28,28,96
    #batch normal layer1                                 
    bn3_loc = BatchNormLayer(conv3_loc, act=tf.identity, is_train=True, name='bn3_loc', trainable=trainable)    
    #pooling layer3(contains prelu): max pooling kernel_size=2, stride=2
    pool3_loc = tl.layers.MaxPool2d(bn3_loc, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='MaxPool_3_2x2')#?,14,14,96
    #prelu
    prelu3_loc = PReluLayer(pool3_loc, name='prelu3_loc')   
    
    #Flatten
    net_shape = prelu3_loc.outputs.get_shape()
    pool3_loc_flat = tl.layers.ReshapeLayer(prelu3_loc, shape=[-1, net_shape[1]*net_shape[2]*net_shape[3]], name='loc_Reshapelayer')#?,18816
    #fully connective layer: 64 output
    fc1_loc = DenseLayer(pool3_loc_flat, n_units=64, W_init=tf.truncated_normal_initializer(mean=0.0,stddev=0.03), name='fc1_loc')#?,64
    #batch normal layer1                                 
    bn_fc1_loc = BatchNormLayer(fc1_loc, act=tf.identity, is_train=True, name='bn_fc1_loc', trainable=trainable)
    #prelu
    prelu_fc1_loc = PReluLayer(bn_fc1_loc, name='prelu_fc1_loc')   
    
    #fully connective layer: 64 output
    fc2_loc = DenseLayer(prelu_fc1_loc, n_units=64, W_init=tf.truncated_normal_initializer(mean=0.0,stddev=0.01), name='fc2_loc')#?,64
    #batch normal layer1                                 
    bn_fc2_loc = BatchNormLayer(fc2_loc, act=tf.identity, is_train=True, name='bn_fc2_loc', trainable=trainable)
    #prelu
    prelu_fc2_loc = PReluLayer(bn_fc2_loc, name='prelu_fc2_loc')
    
    #fully connective layer: 64 output
    fc3_loc = DenseLayer(prelu_fc2_loc, n_units=6, W_init=tf.truncated_normal_initializer(mean=0.0,stddev=0.01), b_init=tf.constant_initializer(np.array([1,0,0,0,1,0],dtype='float32')), name='fc3_loc')#?,6
    transformed_images = tl.layers.SpatialTransformer2dAffineLayer(stn_inputs,fc3_loc,out_size=[112,112])
    
    return transformed_images
                
def resnet(inputs, bottle_neck, blocks, w_init=None, trainable=None, keep_rate=None, scope=None):
    with tf.variable_scope(scope):
        #with tf.variable_scope("spatial_trans"):
        #    spatial_trans_inputs = stn_process_tl(inputs, trainable)
        
        spatial_trans_inputs = tl.layers.InputLayer(inputs, name='input_layer')
        if bottle_neck:
            net = Conv2d(spatial_trans_inputs, n_filter=64, filter_size=(3, 3), strides=(1, 1),
                                   act=None, W_init=w_init, b_init=None, name='conv1', use_cudnn_on_gpu=True)
            net = BatchNormLayer(net, act=tf.identity, name='bn0', is_train=True, trainable=trainable)
            net = PReluLayer(net, name='prelu0')
        else:
            raise ValueError('The standard resnet must support the bottleneck layer')
        for block in blocks:
            with tf.variable_scope(block.scope):
                for i, var in enumerate(block.args):
                    with tf.variable_scope('unit_%d' % (i+1)):
                        net = block.unit_fn(net, depth=var['depth'], depth_bottleneck=var['depth_bottleneck'],
                                            w_init=w_init, stride=var['stride'], rate=var['rate'], scope=None,
                                            trainable=trainable)
        net = BatchNormLayer(net, act=tf.identity, is_train=True, name='E_BN1', trainable=trainable)
        #net = tl.layers.DropoutLayer(net, keep=keep_rate, name='E_Dropout')
        net.outputs = tf.nn.dropout(net.outputs, keep_prob=keep_rate, name='E_Dropout')
        net_shape = net.outputs.get_shape()
        net = tl.layers.ReshapeLayer(net, shape=[-1, net_shape[1]*net_shape[2]*net_shape[3]], name='E_Reshapelayer')
        net = DenseLayer(net, n_units=512, W_init=w_init, name='E_DenseLayer')
        
        #stop_op = net.outputs
        #net.outputs = tf.stop_gradient(stop_op,name='stop_gradient')
        
        net = BatchNormLayer(net, act=tf.identity, is_train=True, fix_gamma=False, trainable=trainable, name='E_BN2')
        return net


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def resnetse_v1_block(scope, base_depth, num_units, stride, rate=1, unit_fn=None):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return Block(scope, unit_fn, [{
      'depth': base_depth,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'rate': rate
  }] + [{
      'depth': base_depth,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'rate': rate
  }] * (num_units - 1))


def get_resnet(inputs, num_layers, type=None, w_init=None, trainable=None, keep_rate=None, sess=None):
    if type == 'ir':
        unit_fn = bottleneck_IR
    elif type == 'se_ir':
        unit_fn = bottleneck_IR_SE
    else:
        raise ValueError('the input fn is unknown')

    if num_layers == 50:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=14, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    elif num_layers == 100:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=13, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=30, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    elif num_layers == 152:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    else:
        raise ValueError('Resnet layer %d is not supported now.' % num_layers)
    net = resnet(inputs=inputs,
                 bottle_neck=True,
                 blocks=blocks,
                 w_init=w_init,
                 trainable=trainable,
                 keep_rate=keep_rate,
                 scope='resnet_v1_%d' % num_layers)
    return net


if __name__ == '__main__':
        x = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_place')
        sess = tf.Session()
        # w_init = tf.truncated_normal_initializer(mean=10, stddev=5e-2)
        w_init = tf.contrib.layers.xavier_initializer(uniform=False)
        # test resnetse
        nets = get_resnet(x, 50, type='ir', w_init=w_init, sess=sess)
        tl.layers.initialize_global_variables(sess)

        for p in tl.layers.get_variables_with_name('W_conv2d', True, True):
            print(p.op.name)
        print('##############'*30)
        with sess:
            nets.print_params()
