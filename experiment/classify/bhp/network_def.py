

from lasagne.layers import Conv1DLayer,\
                           InputLayer, \
                           BatchNormLayer,\
                           DenseLayer, \
                           MaxPool1DLayer, \
                           ConcatLayer, \
                           DropoutLayer

from lasagne.nonlinearities import LeakyRectify,\
                                   sigmoid, \
                                   softmax



#from lasage.regularization import l1, l2

#from lasagne.objectives import squared_error

from lasagne.layers import get_output

import theano.tensor as T

import numpy as np
import theano

def cnn( input_var, input_length, output_length ):

    input_layer = InputLayer( shape = (None, 1, input_length ), \
                        input_var = input_var )

    
    dense_1 = DenseLayer( input_layer, num_units = 10 )

    dense_2 = DenseLayer( dense_1, num_units = 5 )

    dense_3 = DenseLayer( dense_2, num_units = 3 )

    network = DenseLayer( dense_3, num_units = output_length, \
                          nonlinearity = softmax )

    return network

'''
if __name__ == '__main__':

    x = T.tensor3( 'input' )
    
    e,d = auto_encoder( x, 7, 3 )

    xx = np.random.random( 630 ).reshape( 90,1,7 )
    print xx.shape
    y1, y2 = get_output( [d,e], x )


    f = theano.function( [x], [y1, y2], allow_input_downcast = True )

    y11, y22 = f( xx )

    print y11, y22
'''
