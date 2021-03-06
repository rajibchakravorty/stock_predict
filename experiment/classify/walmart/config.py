from os.path import dirname, realpath, join


current_folder = dirname( realpath( __file__ ) )


org = 'walmart'


input_length = 7
output_length = 3

num_epochs = 200
batchsize = 32


learning_rate = 1.0

## ada delta params
rho = 0.95
eps = 1e-6


l1_weight = 1.0
l2_weight = 0.5


start_error = 1e12

model_file = '{0}/ae_org_{1}.npz'.format( current_folder, org )
init_model = None


stat_file = '{0}/learning_stat_{1}.csv'.format( current_folder, org )

