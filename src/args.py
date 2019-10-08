import json
import yaml


def merge(d1, d2):
    # danger: this is recursive and not good for huge structures!
    for k in d2:
        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
            merge(d1[k], d2[k])
        else:
            d1[k] = d2[k]


def get_args(filename='args.yaml'):
    # initialize args with a "complete" set of defaults (?)
    args = yaml.load('defaults.yaml')
    if filename.endswith('.yaml') or filename.endswith('.yml'):
        userargs = yaml.load(filename)
    elif filename.endswith('.json'):
        userargs = json.load(filename)
    merge(args, userargs)
    return args
