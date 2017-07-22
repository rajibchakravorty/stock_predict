
import time
import numpy as np

from lasagne.layers import get_all_param_values


from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


from experiment import train_setup
from import_config import configuration as config

import sys
sys.path.append( '../../utility' )
from util_func import save_epoch_info, iterate_minibatches


from util_func import save_epoch_info, iterate_minibatches


def convert_to_array(X, Y):

    x = np.array( X ).reshape( -1, config.input_length )

    y = np.array( Y ).reshape( -1,  )

    x = np.expand_dims( x, axis = 1 )

    return x, y

def train():

    dt = np.load( config.source_data_file)['arr_0']

    X, Y = zip( *dt )

    sample_size = len( X) 

    train_sample = int( sample_size * 0.8 )
    valid_sample = int( sample_size * 0.1 )

    X_train = X[0:train_sample]
    X_valid = X[train_sample: (train_sample+valid_sample ) ]
    X_test  = X[(train_sample+valid_sample):]

    Y_train = Y[0:train_sample]
    Y_valid = Y[train_sample: (train_sample+valid_sample ) ]
    Y_test  = Y[(train_sample+valid_sample):]

    print 'Train/Valid/Test sample sizes {0}/{1}/{2}'.format( \
                   len( X_train ), len( Y_valid ), len( Y_test ) )


    # get the network
    network, train_fn, valid_fn = train_setup()

    
    start_loss = config.start_loss

    for epoch in range( config.num_epochs ):

        print 'Epoch {0}'.format( epoch+ 1 )

        train_ent_loss = 0.
        train_l1_loss = 0.
        train_l2_loss = 0.
        train_batch = 0
        train_prediction = None
        train_targets = None
        start_time = time.time()

        for batch in iterate_minibatches( X_train, Y_train, config.batchsize, True ):

            x, y = batch
            x, y = convert_to_array( x, y )

            ent, l1, l2, pred = train_fn( x, y )

            train_ent_loss += ent
            train_l1_loss += l1
            train_l2_loss += l2

            #pos_pred = np.reshape( pred[:,1], (-1,1)) 
            target   = np.reshape( y, (-1,1 ) )
            pred_class = np.argmax( pred, axis = 1 )
            pred_class = np.reshape( pred_class, (-1, 1) ) 
            if train_prediction is None:
                train_prediction = pred_class
                train_targets = target
            else:
                train_prediction = np.concatenate( (train_prediction, \
                                                    pred_class ), \
                                                    axis = 0 )
                train_targets = np.concatenate( (train_targets, target ),\
                                                axis = 0 )

            train_batch += 1

        valid_ent_loss = 0.
        valid_batch = 0
        valid_prediction = None
        valid_targets = None
        for batch in iterate_minibatches( X_valid, Y_valid, config.batchsize, False ):

            x, y = batch
            x, y = convert_to_array( x, y )
            
            ent, pred = valid_fn( x, y )
            valid_ent_loss += ent

            target   = np.reshape( y, (-1,1 ) )

            target        = np.reshape( y, (-1,1 ) )
            pred_class  = np.argmax( pred , axis = 1 )
            pred_class  = np.reshape( pred_class, (-1,1))
            if valid_prediction is None:
                valid_prediction = pred_class
                valid_targets = target
            else:
                valid_prediction = np.concatenate( (valid_prediction, \
                                                    pred_class ), \
                                                    axis = 0 )
                valid_targets = np.concatenate( (valid_targets, target ),\
                                                axis = 0 )

            valid_batch += 1

        train_loss = train_ent_loss / train_batch
        train_l1_loss /= train_batch
        train_l2_loss /= train_batch
        valid_loss = valid_ent_loss/valid_batch
        epoch_time = time.time() - start_time

        ##do the performance calculation using sklearn
        train_accuracy = accuracy_score( train_prediction, train_targets )
        valid_accuracy = accuracy_score( valid_prediction, valid_targets )
        print 'Epoch {0} completed in {1}s'.format( epoch+1, epoch_time )

        save_epoch_info( epoch+1, epoch_time, train_loss, train_accuracy, train_l1_loss+train_l2_loss,\
                            valid_accuracy, config.stat_file )

        print 'Epoch stats'
        print ' training accuracy: {0}'.format( train_accuracy )
        print ' validation accuracy: {0}'.format( valid_accuracy )
        print ' l1/l2 {0}/{1}'.format( train_loss, valid_loss )


        if valid_loss <= start_loss:

            np.savez( config.model_file, *get_all_param_values( network ) )
            start_loss = valid_loss
    

if __name__ == '__main__':

    train()
