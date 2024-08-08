import argparse
import torch
import os

parser = argparse.ArgumentParser()

# 1.dataset
parser.add_argument('--dataset', type=str, default='synthetic', help='datasets')
parser.add_argument('--P',type=float,default=0.8)
parser.add_argument('--experiment_name', type=str, default='lp', help='which task')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--input_dim', type=int, default=128, help='dim of input feature')
parser.add_argument('--testlength', type=int, default=3, help='length for test')
parser.add_argument('--num_classes', type=int, default=-1, help='')
parser.add_argument('--nc_layers', type=int, default=2, help='')

# 2 model
parser.add_argument('--hid_dim', type=int, default=32, help='dim of hidden embedding')
parser.add_argument('--weight1', type=float, default=1)
parser.add_argument('--weight2', type=float, default=1)
parser.add_argument('--causal_ratio', type=float, default=0.2)
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.,
                    help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.,
                    help='Temporal attention Dropout (1 - keep probability).')

# 2 attention Architecture params
parser.add_argument('--structural_head_config', type=str, nargs='?', default='4',
                    help='Encoder layer config: # attention heads in each GAT layer')# 8,4
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='32',
                    help='Encoder layer config: # units in each GAT layer') # 32,64
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='4',
                    help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='32',
                    help='Encoder layer config: # units in each Temporal layer')
# 2 ture or false
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--norm',type=int,default=1)
parser.add_argument('--use_RTE', type=int, default=1, help='')
parser.add_argument('--fmask',type=int,default=1)
parser.add_argument('--lin_bias',type=int,default=0)
parser.add_argument('--skip', type=int, default=0, help='') # 1
# 3.experiments
parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train.')
parser.add_argument('--min_epoch', type=int, default=400, help='min epoch')
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate') # 0.002 for nc, motif. 0.01 for lp
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight for L2 loss on basic models.')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate ')# 0.7
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--intervention_times', type=int, default=10, help='intervention times')

# 4 log and others
parser.add_argument('--log_dir', type=str, default="logs/new/")
parser.add_argument('--log_interval', type=int, default=20, help='')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--shift', type=int, default=0)
parser.add_argument('--use_cfg', type=int, default=1)

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

if args.use_cfg:
    if args.dataset == 'collab':
        args.experiment_name = 'lp'
        args.weight1, args.weight2, args.causal_ratio = 1., 1., 0.2
    elif args.dataset == 'act':
        args.experiment_name = 'lp'
        args.weight1, args.weight2, args.causal_ratio = 0.1, 0.1, 0.2
    elif args.dataset == 'Aminer':
        args.experiment_name = 'nc'
        args.lr, args.weight_decay = 0.002, 0.005
        args.weight1, args.weight2, args.causal_ratio = 1e-3, 1e-3, 0.1
        args.hid_dim = 64
        args.structural_head_config, args.structural_layer_config = '8', '64'
        args.temporal_head_config, args.temporal_layer_config = '8', '64'
    elif args.dataset == 'dymotif_data':
        args.experiment_name = 'nc'
        args.lr, args.weight_decay = 0.002, 5e-7
        args.weight1, args.weight2, args.causal_ratio = 1e-3, 1e-3, 0.4
        args.hid_dim = 32
        args.structural_head_config, args.structural_layer_config = '4,4', '32,32'
        args.temporal_head_config, args.temporal_layer_config = '4', '32'
        args.lin_bias = 1
    elif 'synthetic' in args.dataset:
        args.experiment_name = 'lp'
        args.weight1, args.weight2, args.causal_ratio = 1., 1., 0.2
    else:
        raise NotImplementedError(f"dataset {args.dataset} not implemented")
