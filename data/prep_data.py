import pandas as pd

import numpy as np

from os.path import join, dirname, realpath

current_dir = dirname( realpath( __file__ ) )


def prepare_data( org , past_days, future_days ):


    data_file = join( current_dir, '{0}/{0}_close.csv'.format( org ) )


    # 1 for rise, 0 for not rise
    dt = pd.read_csv( data_file, header = 0, delimiter = ',' )
    adjusted_close_price = dt['Adj Close'].values

    data_size = adjusted_close_price.shape[0]
    print 'Total data {0}'.format( data_size )


    X = None
    Y = None
    next_day  = None

    stop_point = data_size - past_days - (future_days -1 )
    for i in range( stop_point ):

        x = adjusted_close_price[i:(i+past_days)]

        if adjusted_close_price[ i+past_days ] > 0:

            y = 1
        else:
            y = 0

        x = np.array( [x] ).reshape(  (1,-1) )

        if (adjusted_close_price[i+past_days] - \
                    adjusted_close_price[i+past_days - 1] ) > 0:
            y = np.array( [1] ).reshape( 1,1 )
        else:
            y = np.array( [0]).reshape( 1,1 )

        n = np.array( adjusted_close_price[(i+past_days):(i+past_days+future_days)] ).reshape( (1,-1) )
        if X is None:

            X = x
            Y = y
            next_day = n
        else:
            X = np.concatenate( (X,x), axis = 0  )
            Y = np.concatenate( (Y,y), axis = 0 )
            next_day = np.concatenate( (next_day, n), axis = 0 )


    return X, Y, next_day

'''
if __name__ == '__main__':

    org = 'walmart'
    org_file = join( org, '{0}_close.csv'.format( org ) )

    #X, Y, next_day = classification_data( 7 )
    X, Y, next_day = prepare_data( org_file, 7, 3)

    print X.shape, Y.shape

    print X[-1], Y[-1], next_day[-1]

    assert X.shape[0] == Y.shape[0]

    print 'Total prepared data {0}'.format( X.shape[0] )

    print np.sum( Y )
'''
