

org = 'bhp'


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

model_file = 'ae_org_{0}.npz'.format( org )
init_model = None


stat_file = 'learning_stat_{0}.csv'.format( org )

