import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
import os
import cPickle
import sys
import utils

import Tree


class FOL(object):

    def __init__(self, model, train_iter = 10000, batch_size = 64, gamma = 0.5, eta = 0.0001, 
                 mnist_dir='./data/mnist', log_dir='./logs', model_save_path='./model'):
        
	
        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        
	#Tree parameters
	self.gamma = gamma
	self.eta = eta
	
	self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	
    def load_mnist(self, image_dir, split='train'):
        print ('Loading MNIST dataset.')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = cPickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        return images, labels
	
    def train(self):

        train_images, train_labels = self.load_mnist(self.mnist_dir, split='train')
        test_images, test_labels = self.load_mnist(self.mnist_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    # defining the tree, with m as the number of training images
	    tree = Tree.Tree(m = len(train_images))
	    
	    # initializing the tree
	    tree.initialize()
	    
	    for t in range(self.train_iter):
		
		#sampling from the tree
		(i, p_i) = tree.sample(gamma = self.gamma)
		sampled_image = np.expand_dims(train_images[i],0)
		sampled_label = np.expand_dims(train_labels[i],0)
           
		#minimizing the loss associated with the sampled image/label, using Adam as OLA (see paper)
		feed_dict = {model.images: sampled_image, model.labels: sampled_label} 
	        _, l = sess.run([model.train_op, model.loss], feed_dict) 

		#updating the tree
		tree.update(i, np.exp(self.eta*l/p_i))


		#evaluate the model
		if (t+1) % 10 == 0:
		    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
		    
		    train_rand_idxs = np.random.permutation(train_images.shape[0])[:1000]
		    test_rand_idxs = np.random.permutation(test_images.shape[0])[:1000]
		    
		    train_acc, train_loss = sess.run(fetches=[model.accuracy, model.loss], 
					   feed_dict={model.images: train_images[train_rand_idxs], 
						      model.labels: train_labels[train_rand_idxs]})
		    test_acc, test_loss = sess.run(fetches=[model.accuracy, model.loss], 
					   feed_dict={model.images: test_images[test_rand_idxs], 
						      model.labels: test_labels[test_rand_idxs]})
		    				      
						      
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d] train_loss: [%.4f] train_acc: [%.4f] test_loss: [%.4f] test_acc: [%.4f]'%(t+1, self.train_iter, train_loss, train_acc, test_loss, test_acc))
		    
		#~ if (t+1) % 100 == 0:
		    #~ saver.save(sess, os.path.join(self.model_save_path, 'encoder'))
	    
    def test(self):
	
	raise Exception('To be implemented.')
	

      
if __name__=='__main__':

    print 'To be implemented.'
