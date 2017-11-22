import tensorflow as tf
from Model import Model
from FOL import FOL

import os

import numpy.random as npr

flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "directory for saving the model")
FLAGS = flags.FLAGS

def main(_):
    
    GPU_ID = FLAGS.gpu

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    learning_rate = 0.001
    
    train_iter = 100000
    batch_size = 64 #useless at this point
    gamma = 0.5
    eta = 0.0001 #check, not sure
    

    model = Model(learning_rate=learning_rate)
    fol = FOL(model, train_iter = train_iter, batch_size = batch_size, gamma = gamma, eta=eta, model_save_path='./model')
    
    fol.train()
    
        
if __name__ == '__main__':
    tf.app.run()



    


