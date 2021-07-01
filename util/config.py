import copy, json
from collections import OrderedDict

import numpy as np

import torch
from torch import nn

from transformers import BertConfig, BertTokenizer, BertModel, AdamW

from .dataset import BaseDataset, SentSimDataset, EntlmntDataset, OntoDataset
from . import common as C
from . import modules as M
from . import processor as P


RGRSN_LOSS_MAP = {'contrastive':M.ContrastiveLoss}
NORM_TYPE_MAP = {'batch':nn.BatchNorm1d, 'layer':nn.LayerNorm}
ACTVTN_MAP = {'relu':nn.ReLU, 'sigmoid':nn.Sigmoid, 'tanh':nn.Tanh}
SIM_FUNC_MAP = {'sim':'sim', 'dist':'dist'}
NUM_UNDERLINE_IN_ORIG = {'biolarkgsc':0, 'copd':1}


### Serializable interface ###
class Serializable(object):
    def __init__(self):
        raise NotImplementedError

    def to_yaml(self, fpath='config.yaml', pkl_fpath='config.pkl'):
        from bionlp.util.io import write_yaml
        write_yaml(self.serialize(pkl_fpath=pkl_fpath, skip_paths=self.skip_paths if hasattr(self, 'skip_paths') and self.skip_paths is not None else []), fpath=fpath)

    def to_json(self, fpath='config.json', pkl_fpath='config.pkl'):
        from bionlp.util.io import write_json
        write_json(self.serialize(pkl_fpath=pkl_fpath, skip_paths=self.skip_paths if hasattr(self, 'skip_paths') and self.skip_paths is not None else []), fpath=fpath)

    def to_file(self, fpath='config.json', pkl_fpath='config.pkl'):
        if fpath.endswith('.yaml') or fpath.endswith('.yml'):
            self.to_yaml(fpath=fpath, pkl_fpath=pkl_fpath)
        elif fpath.endswith('.json'):
            self.to_json(fpath=fpath, pkl_fpath=pkl_fpath)


    def serialize(self, pkl_fpath='config.pkl', skip_paths=[]):
        def _rec_serialize(data, path='', external_dict={}, skip_paths=[]):
            if path in skip_paths:
                if self.verbose: logging.debug('Skip serializing attribute path [%s]' % path)
                return ''
            try:
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                if type(data) is dict:
                    for k, v in data.items():
                        data[k] = _rec_serialize(v, path='/'.join([path, k]), external_dict=external_dict, skip_paths=skip_paths)
                elif type(data) is tuple or type(data) is list or type(data) is set:
                    new_data = []
                    for i, x in enumerate(data):
                        new_data.append(_rec_serialize(x, path='/'.join([path, str(i)]), external_dict=external_dict, skip_paths=skip_paths))
                    data = new_data
                else:
                    external_dict[path] = data
                    data = path
            return data
        from bionlp.util.io import write_obj
        attributes = dict([(k,copy.deepcopy(v)) for k, v in self.__dict__.items() if not k.startswith('_') and (not callable(v) or not hasattr(v, '__self__'))])
        pkl_obj, skip_paths = {}, set(skip_paths)
        serialized_attrs = _rec_serialize(attributes, path='', external_dict=pkl_obj, skip_paths=skip_paths)
        if pkl_fpath is not None: write_obj(pkl_obj, fpath=pkl_fpath)
        return serialized_attrs

# Config Class
class SimpleConfig(Serializable):
    def __init__(self, attributes, pkl_fpath='config.pkl', import_lib=False, import_map={}):
        def _rec_deserialize(data, path='', external_dict={}):
            if type(data) is dict:
                for k, v in data.items():
                    data[k] = _rec_deserialize(v, path='/'.join([path, k]), external_dict=external_dict)
            elif type(data) is tuple or type(data) is list or type(data) is set:
                new_data = []
                for i, x in enumerate(data):
                    new_data.append(_rec_deserialize(x, path='/'.join([path, str(i)]), external_dict=external_dict))
                data = new_data
            elif import_lib and type(data) is str and data.startswith('import://'):
                import importlib
                module_str, cls_func = data[len('import://'):].split('/')
                module = importlib.import_module(import_map.setdefault(module_str, module_str))
                data = module if cls_func is None else getattr(module, cls_func)
            return data
        try:
            self.pkl_obj = C.read_obj(pkl_fpath)
        except Exception as e:
            self.pkl_obj = {}
        self.skip_paths = attributes.setdefault('skip_paths', [])
        self.skip_paths = (self.skip_paths if type(self.skip_paths) is list else []) + ['/pkl_obj', '/importlibs', '/skip_paths']
        del attributes['skip_paths']
        for k, v in _rec_deserialize(attributes, path='', external_dict=self.pkl_obj).items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return self.__dict__.setdefault(name, self.__dict__['_'+name] if '_'+name in self.__dict__ else None)

    def to_file(self, fpath='config.json'):
        super(SimpleConfig, self).to_file(fpath=fpath, pkl_fpath=None)

    @classmethod
    def from_yaml(cls, fpath='config.yaml', pkl_fpath='config.pkl', import_lib=False, import_map={}, updates={}):
        attributes = C.read_yaml(fpath)
        C.exhausted_updates(attributes, updates)
        return cls(attributes, pkl_fpath=pkl_fpath, import_lib=import_lib, import_map=import_map)

    @classmethod
    def from_json(cls, fpath='config.json', pkl_fpath='config.pkl', import_lib=False, import_map={}, updates={}):
        attributes = C.read_json(fpath)
        C.exhausted_updates(attributes, updates)
        return cls(attributes, pkl_fpath=pkl_fpath, import_lib=import_lib, import_map=import_map)

    @classmethod
    def from_file(cls, fpath='config.json', pkl_fpath='config.pkl', import_lib=False, import_map={}, updates={}):
        if fpath.endswith('.yaml') or fpath.endswith('.yml'):
            return cls.from_yaml(fpath, pkl_fpath=pkl_fpath, import_lib=import_lib, import_map=import_map, updates=updates)
        elif fpath.endswith('.json'):
            return cls.from_json(fpath, pkl_fpath=pkl_fpath, import_lib=import_lib, import_map=import_map, updates=updates)

    @classmethod
    def from_file_importmap(cls, fpath='config.json', pkl_fpath='config.pkl', import_lib=False, import_map_fpath='import_map.json', updates={}):
        try:
            import_map = C.read_json(import_map_fpath)
        except Exception as e:
            import_map = {}
        return cls.from_file(fpath=fpath, pkl_fpath=pkl_fpath, import_lib=import_lib, import_map=import_map, updates=updates)


### Unit Test ###
def test_config():
    config = SimpleConfig.from_file('config.json')
    print(config.__dict__)
