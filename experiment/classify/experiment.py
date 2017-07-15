

from theano import function

import theano.tensor as T


from lasagne.regularization import regularize_network_params, l1, l2

from lasagne.objectives import squared_error

from lasagne.layers import get_output, get_all_params, \
                           get_all_param_values, \
                           set_all_param_values, \
                           count_params

from lasagne.updates import adadelta

from import_config import configuration as config
from network_def import auto_encoder as cnn

import sys
sys.path.append( '../../data' )
from prep_data import prepare_data



def test_setup():

    x = T.tensor3( 'input' )
    y = T.matrix( 'output' )

    encoding, decoding = cnn( x, config.input_length, config.output_length )


    print 'Loading parameters'


    with np.load( config.init_model ) as f:
        param_values = [ f['arr_%d' % i] for i in range( len( f.files ) ) ]

    set_all_param_values( decoding, param_values )
    prediction = get_output( decoding, deterministic = True )

    error = squared_error( y, prediction )       

    test_fn = function( [x], [prediction, error], allow_input_downcast = True )

    return test_fn

def train_setup():


    x = T.tensor3( 'input' )
    y = T.matrix( 'output' )

    encoding, decoding = cnn( x, config.input_length, config.output_length )


    print 'Number of Parameters {0}'.format( count_params( decoding ) )


    if config.init_model is not None:

        with np.load( config.init_model ) as f:
            param_values = [ f['arr_%d' % i] for i in range( len( f.files ) ) ]

        set_all_param_values( decoding, param_values )

    # training tasks in sequence

    prediction = get_output( decoding )

    error = squared_error( y, prediction )
    error = error.mean()

    l1_norm = config.l1_weight * regularize_network_params( decoding, l1 )
    l2_norm = config.l2_weight * regularize_network_params( decoding, l2 )

    total_error = error + l1_norm + l2_norm

    params = get_all_params( decoding, trainable = True )

    updates = adadelta( total_error, params, config.learning_rate, \
                                             config.rho, \
                                             config.eps )

    train_fn = function( [x, y], [error, l1_norm, l2_norm], \
                              updates = updates, \
                              allow_input_downcast = True )



    

    val_prediction = get_output( decoding, deterministic = True )
    val_error      = squared_error( y, val_prediction )
    val_error      = val_error.mean()

    val_fn         = function( [x,y], val_error, allow_input_downcast = True )

    return encoding, decoding, train_fn, val_fn
