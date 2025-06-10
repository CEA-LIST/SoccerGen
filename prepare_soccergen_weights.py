import torch
from argparse import ArgumentParser
from cldm.model import load_state_dict

parser = ArgumentParser()
parser.add_argument('--base_controlnet_state_dict', type=str)
parser.add_argument('--dest_path', type=str)
args = parser.parse_args()

state_dict = load_state_dict(args.base_controlnet_state_dict, location='cpu')

new_state_dict = dict()
for key in state_dict.keys():
    if not key.startswith('control_model.input_hint_block'):
        new_state_dict[key] = state_dict[key]

torch.save(new_state_dict, args.dest_path)