import torch
from collections import OrderedDict
from os import path as osp
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Download the model provided by mmflow and convert the model state dict which can be loaded in SCFlow project')
    parser.add_argument('--model_url', type=str, )
    args = parser.parse_args()
    return args






if __name__ == '__main__':
    args = get_args()
    model = torch.hub.load_state_dict_from_url(
        args.model_url,
        map_location='cpu')
    state_dict = model['state_dict']
    new_model = dict(
        meta = model['state_dict'],
        optimizer=model['optimizer'],
    )
    new_state_dict=OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('encoder'):
            new_state_dict[k.replace('encoder', 'real_encoder')] = v
            new_state_dict[k.replace('encoder', 'render_encoder')] = v
        else:
            new_state_dict[k] = v
    new_model['state_dict'] = new_state_dict
    model_name = args.model_url.split('/')[-1]
    model_name = model_name.split('.')[0] + '_convertered.pth'
    save_path = osp.join('work_dirs', model_name)
    print(f'Successfully Convert Model! Save model in {save_path}')
    torch.save(new_model, save_path)
            
