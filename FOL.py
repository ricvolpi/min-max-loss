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

    def __init__(model, train_iter = train_iter, batch_size = batch_size, gamma = gamma, model_save_path='./model', 
                 mnist_dir='./data/mnist', log_dir='./logs', model_save_path='./model', model='encoder'):
        
	
        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        
	#Tree parameters
	self.gamma = gamma
	self.eta = learning_rate #? check!!!
	
	self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	
    def load_mnist(self, image_dir, split='train'):
        print ('Loading MNIST dataset.')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
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
	    
	    for t in self.train_iter:
		
		#sampling from the tree
		(i, p_i) = tree.sample(gamma = self.gamma)
		sampled_image = train_images[i]
		sampled_label = train_labels[i]
           
		#minimizing the loss associated with the sampled image/label, using Adam as OLA (see paper)
		feed_dict = {model.images: sampled_image, model.labels: sampled_label} 
	        _, l = sess.run([model.train_op, model.loss], feed_dict) 

		#updating the tree
		tree.update(i, np.exp(eta*l/p_i))


		#evaluate the model
		if (t+1) % 100 == 0:
		    summary, l, src_acc = sess.run([model.summary_op, model.loss, model.src_accuracy], feed_dict)
		    src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:1000]
		    trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:1000]
		    test_src_acc, test_trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
					   feed_dict={model.src_images: src_test_images[src_rand_idxs], 
						      model.src_labels: src_test_labels[src_rand_idxs],
						      model.trg_images: trg_test_images[trg_rand_idxs], 
						      model.trg_labels: trg_test_labels[trg_rand_idxs]})
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] src test acc [%.2f] trg test acc [%.2f]' \
			       %(t+1, self.train_iter, l, src_acc, test_src_acc, test_trg_acc))
		    
		if (t+1) % 100 == 0:
		    saver.save(sess, os.path.join(self.model_save_path, 'model'))
	    
    def test(self):
	
	train_images, train_labels = self.load_mnist(self.mnist_dir, split='train')
	test_images, test_labels = self.load_mnist(self.mnist_dir, split='test')

	# build a graph
	model = self.model
	model.build_model()
	
	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
	with tf.Session(config=self.config) as sess:
	    tf.global_variables_initializer().run()
	    saver = tf.train.Saver()
	    
	    t = 0
	    
	    acc = []
	    
	    while(True):
		
		print ('Loading test model.')
		variables_to_restore = slim.get_model_variables(scope='encoder')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, os.path.join(self.model_save_path, 'model'))
		

	    
		t+=1
    
		src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:]
		trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:]
		test_src_acc, test_trg_acc, trg_pred = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.trg_pred], 
				       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
						  model.src_labels: src_test_labels[src_rand_idxs],
						  model.trg_images: trg_test_images[trg_rand_idxs], 
						  model.trg_labels: trg_test_labels[trg_rand_idxs]})
		src_acc = sess.run(model.src_accuracy, feed_dict={model.src_images: src_images[:10000], 
								  model.src_labels: src_labels[:10000],
						                  model.trg_images: trg_test_images[trg_rand_idxs], 
								  model.trg_labels: trg_test_labels[trg_rand_idxs]})
						  
		print ('Step: [%d/%d] src train acc [%.3f]  src test acc [%.3f] trg test acc [%.3f]' \
			   %(t+1, self.pretrain_iter, src_acc, test_src_acc, test_trg_acc))
		
		print confusion_matrix(trg_test_labels[trg_rand_idxs], trg_pred)	   
		
		acc.append(test_trg_acc)
		with open(self.protocol + '_' + algorithm + '.pkl', 'wb') as f:
		    cPickle.dump(acc,f,cPickle.HIGHEST_PROTOCOL)
      
if __name__=='__main__':

    print 'To be implemented.'
