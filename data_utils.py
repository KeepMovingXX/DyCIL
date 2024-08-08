import os
import torch

def load_data(args):
    dataset = args.dataset
    if dataset == 'collab':
        dataroot = os.path.join('./data/',dataset)
        processed_datafile = f"{dataroot}/processed2"
        args.testlength = 5
        args.vallength = 1
        args.length = 16
        args.split = 0
        data = torch.load(f'{processed_datafile}-{args.split}')
        args.input_dim=data['x'].shape[1]
        args.num_nodes = len(data['x'])

    elif dataset == 'Aminer':
        dataroot = os.path.join('./data/',dataset)
        processed_datafile = f"{dataroot}/processed_data_128.pt"
        data = torch.load(processed_datafile)
        args.testlength = 3
        args.vallength = 3
        args.length = 17
        args.input_dim=data['x'].shape[1]
        args.num_nodes = len(data['x'])
        args.num_classes = max(data['y']).item() + 1

    elif dataset == 'act':
        dataroot = os.path.join('./data/',dataset)
        data = torch.load(dataroot)
        args.testlength = 8
        args.vallength = 2
        args.length = 30
        args.split = 0
        args.input_dim = data['x'].shape[1]
        args.num_nodes = len(data['x'])

    elif dataset == 'dymotif_data':
        dataroot = os.path.join('./data/DyMotif-0.4', 'raw', dataset)
        data = torch.load(dataroot)
        args.testlength = 3
        args.vallength = 3
        args.length = 30
        args.num_classes = 3
        args.input_dim = data['x_list'][-1].shape[1]
        args.num_nodes = data['x_list'][-1].shape[0]

    elif dataset == 'synthetic':
        dataroot = os.path.join('./data/',dataset)
        processed_datafile = f"{dataroot}/processed2"
        if args.P == 0.4:
            data = torch.load(f'{processed_datafile}-sythetic2-{0.4, 0.05, 0.1, 0.0}')
        elif args.P == 0.6:
            data = torch.load(f'{processed_datafile}-sythetic2-{0.6, 0.05, 0.1, 0.0}')
        else:
            data = torch.load(f'{processed_datafile}-sythetic2-{0.8, 0.05, 0.1, 0.0}')
        args.testlength = 5
        args.vallength = 1
        args.length = 16
        args.split = 0
        args.input_dim = data['x'][0].shape[1]
        args.num_nodes = len(data['x'][0])

    return args, data

