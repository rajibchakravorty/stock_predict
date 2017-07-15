
from theano import function

import theano.tensor as T

from lasagne.layers import get_output, get_all_params, \
                           get_all_param_values, \
                           set_all_param_values, \
                           count_params

from lasagne.updates import adadelta
from lasagne.objectives import squared_error


from import_config import configuration as config
from network_def import auto_encoder as cnn



def get_ae_fn( include_dec = None  ):

    x = T.tensor3( 'input' )

    encoding, decoding = cnn( x, config.input_length, config.output_length )


    print 'Loading parameters'


    with np.load( config.init_model ) as f:
        param_values = [ f['arr_%d' % i] for i in range( len( f.files ) ) ]

    set_all_param_values( decoding, param_values )
    prediction = get_output( encoding, deterministic = True )

    encode_fn = function( [x], prediction, allow_input_downcast = True )

    decode_fn = None
    if include_dec:
        y = T.matrix( 'output' )
        error = squared_error( y, prediction )

        decode_fn = function( [x,y], [prediction, error], \ 
                              allow_input_downcast = True )

    return encode_fn, decode_fn

                                                                                                                                                                          1,0-1         Top
def get_decoder_fn( ):

    x = T.tensor3( 'input' )

    encoding, decoding = cnn( x, config.input_length, config.output_length )


    print 'Loading parameters'


    with np.load( config.init_model ) as f:
        param_values = [ f['arr_%d' % i] for i in range( len( f.files ) ) ]

    set_all_param_values( decoding, param_values )
    prediction = get_output( encoding, deterministic = True )

    encode_fn = function( [x], prediction )

    return encode_fn


