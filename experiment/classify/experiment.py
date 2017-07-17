

from theano import function


import theano.tensor as T


from lasagne.regularization import regularize_network_params, l1, l2

from lasagne.objectives import categorical_crossentropy

from lasagne.layers import get_output, get_all_params, \
                           get_all_param_values, \
                           set_all_param_values, \
                           count_params

from lasagne.updates import adadelta

from import_config import configuration as config
from network_def import cnn as cnn



def test_setup():

    x = T.tensor3( 'input' )
    y = T.lvector( 'output' )

    network = cnn( x, config.input_length, config.output_length )


    print 'Loading parameters'


    with np.load( config.model_file ) as f:
        param_values = [ f['arr_%d' % i] for i in range( len( f.files ) ) ]

    set_all_param_values( network, param_values )
    prediction = get_output( network, deterministic = True )

    ent = categorical_crossentropy( prediction,y )

    test_fn = function( [x,y], [prediction, ent], allow_input_downcast = True )

    return test_fn

def train_setup():


    x = T.tensor3( 'input' )
    y = T.lvector( 'output' )

    network = cnn( x, config.input_length, config.output_length )


    print 'Number of Parameters {0}'.format( count_params( network ) )

    if config.init_model is not None:

        with np.load( config.init_model ) as f:
            param_values = [ f['arr_%d' % i] for i in range( len( f.files ) ) ]

        set_all_param_values( decoding, param_values )

    # training tasks in sequence

    prediction = get_output( network )

    ent = categorical_crossentropy( prediction,y )
    ent = ent.mean()

    l1_norm = config.l1_weight * regularize_network_params( network, l1 )
    l2_norm = config.l2_weight * regularize_network_params( network, l2 )

    total_error = ent + l1_norm + l2_norm

    params = get_all_params( network, trainable = True )

    updates = adadelta( total_error, params, config.learning_rate, \
                                             config.rho, \
                                             config.eps )

    train_fn = function( [x, y], [ent, l1_norm, l2_norm], \
                              updates = updates, \
                              allow_input_downcast = True )


    val_prediction = get_output( network, deterministic = True )
    val_ent        = categorical_crossentropy( val_prediction, y )
    val_ent        = val_ent.mean()

    val_fn         = function( [x,y], val_ent, allow_input_downcast = True )

    return network, train_fn, val_fn
