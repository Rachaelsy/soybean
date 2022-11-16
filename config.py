rand_seed = 314

model_name = 'LSTM'  # ['RNN', 'GRU', 'LSTM', 'TCN', 'STCN']
device = 'cpu'  # 'cpu' or 'cuda'
input_size = 4
hidden_size = 80
output_size = 1
num_layers = 4
levels = 4
kernel_size = 4
dropout = 0.25
in_channels = 18

batch_size = 1
lr = 0.001
n_epochs = 101


def print_para():
    print('\n------ Parameters ------')
    print('rand_seed = {}'.format(rand_seed))
    print('device = {}'.format(device))
    print('input_size = {}'.format(input_size))
    print('hidden_size = {}'.format(hidden_size))
    print('num_layers = {}'.format(num_layers))
    print('output_size = {}'.format(output_size))
    print('levels (for TCN) = {}'.format(levels))
    print('kernel_size (for TCN) = {}'.format(kernel_size))
    print('dropout (for TCN) = {}'.format(dropout))
    print('in_channels (for STCN) = {}'.format(in_channels))
    print('batch_size = {}'.format(batch_size))
    print('lr = {}'.format(lr))
    print('n_epochs = {}'.format(n_epochs))
    print('------------------------\n')

