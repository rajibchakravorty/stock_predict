
import time
import numpy as np

from lasagne.layers import get_all_param_values


from sklearn.utils import shuffle

from experiment import train_setup
from import_config import configuration as config




import sys
sys.path.append( '../../data' )
from prep_data import prepare_data

sys.path.append( '../../utility' )
from util_func import save_epoch_info, iterate_minibatches


from util_func import save_epoch_info, iterate_minibatches


def convert_to_array(X, Y):

    x = np.array( X ).reshape( -1, config.input_length )

    y = np.array( Y ).reshape( -1, config.output_length )

    x = np.expand_dims( x, axis = 1 )

    return x, y

def train():

    X, _ , next_days = prepare_data( config.org, \
                                     config.input_length, \
                                     config.output_length )

    X, N = shuffle( X, next_days, random_state = 10 )
    

    X = list( X )
    N = list( N )

    sample_size = len( X )

    train_sample = int( sample_size * 0.8 )
    valid_sample = int( sample_size * 0.1 )

    X_train = X[0:train_sample]
    X_valid = X[train_sample: (train_sample+valid_sample ) ]
    X_test  = X[(train_sample+valid_sample):]

    N_train = N[0:train_sample]
    N_valid = N[train_sample: (train_sample+valid_sample ) ]
    N_test  = N[(train_sample+valid_sample):]

    print 'Train/Valid/Test sample sizes {0}/{1}/{2}'.format( \
                   len( X_train ), len( N_valid ), len( N_test ) )


    # get the network
    encoder, decoder, train_fn, valid_fn = train_setup()

    
    start_error = config.start_error

    for epoch in range( config.num_epochs ):

        print 'Epoch {0}'.format( epoch+ 1 )

        train_sq_error = 0.
        train_l1_loss = 0.
        train_l2_loss = 0.
        train_batch = 0
        start_time = time.time()

        for batch in iterate_minibatches( X_train, N_train, config.batchsize, True ):

            x, y = batch
            x, y = convert_to_array( x, y )

            error, l1, l2 = train_fn( x, y )

            train_sq_error += error
            train_l1_loss += l1
            train_l2_loss += l2

            train_batch += 1

        valid_sq_error = 0.
        valid_batch = 0
        for batch in iterate_minibatches( X_valid, N_valid, config.batchsize, False ):

            x, y = batch
            x, y = convert_to_array( x, y )
            
            error = valid_fn( x, y )

            valid_sq_error += error
            valid_batch += 1

        train_error = train_sq_error / train_batch
        train_l1_loss /= train_batch
        train_l2_loss /= train_batch
        valid_error = valid_sq_error/valid_batch
        epoch_time = time.time() - start_time
        print 'Epoch {0} completed in {1}s'.format( epoch+1, epoch_time )

        save_epoch_info( epoch+1, epoch_time, train_error, train_l1_loss, train_l2_loss,\
                            valid_error, config.stat_file )

        print 'Epoch stats'
        print ' training squared error: {0}'.format( train_error )
        print ' validation squared error: {0}'.format( valid_error )
        print ' l1/l2 {0}/{1}'.format( train_l1_loss, train_l2_loss )

        if valid_sq_error <= start_error:

            np.savez( config.model_file, *get_all_param_values( decoder ) )
            start_error = valid_sq_error
    

if __name__ == '__main__':

    train()

