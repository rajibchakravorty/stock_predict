
import numpy as np

import random



def save_epoch_info( epoch, epoch_time, square_loss, l1, l2,\
                            val_loss, filename ):

    f = open( filename, 'a' )

    f.write( '{0},{1},{2},{3},{4},{5}\n'.format( epoch, epoch_time,\
                                               square_loss, val_loss, \
                                               l1, l2  ) )

    f.close()


def iterate_minibatches( inputs, target, batchsize, shuffle = False ):

    if shuffle:

        combined = list( zip( inputs, target ) )
        random.shuffle( combined)
        inputs, target = zip( *combined )



    for start_idx in range( 0, len( inputs ), batchsize ):

        end_idx = start_idx + batchsize

        if end_idx >= len( inputs ):

            end_idx = len( inputs )
        #print type( inputs ), type( target )
        yield inputs[start_idx:end_idx], target[start_idx:end_idx]


