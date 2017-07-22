from os.path import dirname, realpath, join


current_folder = dirname( realpath( __file__ ) )


org = 'bhp'

# this is a linked file -> ../../ae/{org}/encoded_values_{org}.npz
source_data_file = '{0}/encoded_values.npz'.format( current_folder )

input_length = 50
output_length = 2

num_epochs = 200
batchsize = 32


learning_rate = 1.0

## ada delta params
rho = 0.95
eps = 1e-6


l1_weight = 0.1
l2_weight = 0.5


start_loss = 1e12

model_file = '{0}/predict_org_{1}.npz'.format( current_folder, org )
init_model = None


stat_file = '{0}/learning_stat_{1}.csv'.format( current_folder, org )

