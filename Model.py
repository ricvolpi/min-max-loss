import tensorflow as tf
import tensorflow.contrib.slim as slim

class Model(object):
    """Neural network
    """
    def __init__(self, mode='train', learning_rate=0.0003):
        self.learning_rate = learning_rate
	   
    def encoder(self, images, reuse=False):
	
	with tf.variable_scope('encoder', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
		    net = slim.conv2d(images, 64, 5, scope='conv1')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
		    net = slim.conv2d(net, 128, 5, scope='conv2')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.fully_connected(net, 1024, scope='fc3')
		    net = slim.fully_connected(net, 1024, scope='fc4')
		    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
		    return net

    def build_model(self):
        
	self.images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'images')
	self.labels = tf.placeholder(tf.int64, [None], 'labels')
	
	self.logits = self.encoder(self.images)
	    
	self.pred = tf.argmax(self.logits, 1)
	self.correct_pred = tf.equal(self.pred, self.labels)
	self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
	
	self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
	self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
	self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	
	loss_summary = tf.summary.scalar('classification_loss', self.loss)
	accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
	self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])
	
