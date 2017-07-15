
import time
import numpy as np

from lasagne.layers import get_all_param_values



from sklearn.utils import shuffle

from run_ae_model import get_ae_fn
from import_config import configuration as config

import sys
sys.path.append( '../../data' )
from prep_data import prepare_data




def convert_to_array(X, Y):

    x = np.array( X ).reshape( -1, config.input_length )

    y = np.array( Y ).reshape( -1, config.output_length )

    x = np.expand_dims( x, axis = 1 )

    return x, y

def encoding():

    X, Y , _ = prepare_data( config.org, \
                                     config.input_length, \
                                     config.output_length )

    print 'Sample size {0}'.format( X.shape[0] )


    # get the network
    encoder, _ = get_ae_fn()

    X = np.expand_dims( X, axis = 1 )
    encoded_values = encoder( X )

    np.savez( config.ae_encoder_file, zip(encoded_values, Y ) )   

if __name__ == '__main__':

    encoding()

