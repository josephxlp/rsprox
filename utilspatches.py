
import os 
import yaml
import time 
import pickle

def write_yaml(yaml_data, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
    print('write_yaml')


def read_yaml(filename):
    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        print('read_yaml')
        return yaml_data

def filter_dictionary_by_keys(dictionary, key_list):
    filtered_dict = {key: value for key, value in dictionary.items() if key in key_list}
    return filtered_dict

def split_dictionary_pair(data_dict):
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    return keys, values

def load_variables(i_wdir, tile, ps):
    yaml_tile = f"{i_wdir}/{tile}/{tile}_ds.yaml"
    print(yaml_tile)
    wdir = f"{i_wdir}_patches{ps}"
    tilename = yaml_tile.split('/')[-2]
    return yaml_tile, wdir, tilename

def load_patch_params(yaml_tile, wdir, tilename, vrtvars):
    os.makedirs(wdir,exist_ok=True)
    data_dict = read_yaml(yaml_tile)
    # print(list(data_dict.keys()) == vrtvars)
    wdir_tile_dpath = os.path.join(wdir, tilename)
    os.makedirs(wdir_tile_dpath,exist_ok=True)
    data_dict_filter = filter_dictionary_by_keys(data_dict, vrtvars)
    names, paths = split_dictionary_pair(data_dict_filter)
    assert len(names) == len(paths), 'len(names) == len(paths) failed!!!'
    return names, paths, wdir_tile_dpath

def filter_patch_params(names, paths, wdir_tile_dpath, i):
    iname = names[i]
    ipath = paths[i]
    wdir_tile_dpath_var = os.path.join(wdir_tile_dpath, iname)
    os.makedirs(wdir_tile_dpath_var, exist_ok=True)
    return iname, ipath, wdir_tile_dpath_var
