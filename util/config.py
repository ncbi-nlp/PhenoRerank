import copy
from collections import OrderedDict

import numpy as np

import torch
from torch import nn

from transformers import BertConfig, BertTokenizer, BertModel, AdamW

from .dataset import BaseDataset, SentSimDataset, EntlmntDataset
from . import modules as M
from . import processor as P


### Module Class and Parameter Config ###
NORM_TYPE_MAP = {'batch':nn.BatchNorm1d, 'layer':nn.LayerNorm}
ACTVTN_MAP = {'relu':nn.ReLU, 'sigmoid':nn.Sigmoid, 'tanh':nn.Tanh}
RGRSN_LOSS_MAP = {'contrastive':M.ContrastiveLoss}


### Universal Config Class ###
class Configurable(object):
    # Task related parameters
    TASK_CONFIG_TEMPLATE_KEYS = ['task_type', 'task_ds', 'task_col', 'task_trsfm', 'task_ext_trsfm', 'task_ext_params']
    TASK_CONFIG_TEMPLATE_VALUES = {
        'mltc-base': ['mltc-clf', BaseDataset, {'index':'id', 'X':'text', 'y':'label'}, (['_mltc_transform'], [{}]), ([P._trim_transform, P._sentclf_transform, P._pad_transform], [{},{},{}]), {'mdlcfg':{'maxlen':128}, 'ignored_label':'false'}],
        'entlmnt-base': ['entlmnt', EntlmntDataset, {'index':'id', 'X':['sentence1','sentence2'], 'y':'label'}, (['_mltc_transform'], [{}]), ([P._trim_transform, P._entlmnt_transform, P._pad_transform], [{},{},{}]), {'mdlcfg':{'sentsim_func':None, 'concat_strategy':None, 'maxlen':128}}]
    }
    TASK_CONFIG_TEMPLATE_DEFAULTS = TASK_CONFIG_TEMPLATE_VALUES['mltc-base']
    PREDEFINED_TASK_CONFIG_KEYS = ['template', 'task_path', 'kwargs']
    PREDEFINED_TASK_CONFIG_VALUES = {
        'biolarkgsc': ['mltl-base', 'biolarkgsc', {}],
		'biolarkgsc_entilement': ['entlmnt-base', 'biolarkgsc.entlmnt', {'task_col':{'index':'id', 'X':['text1', 'text2'], 'y':'label', 'ontoid':'label2'}}],
    }
    # Model related parameters
    MODEL_CONFIG_TEMPLATE_KEYS = ['encode_func', 'clf_ext_params', 'optmzr']
    MODEL_CONFIG_TEMPLATE_VALUES = {
        'transformer-base': [P._base_encode, {'lm_loss':False, 'fchdim':0, 'iactvtn':'relu', 'oactvtn':'sigmoid', 'pdrop':0.2, 'do_norm':True, 'norm_type':'batch', 'do_extlin':False, 'do_lastdrop':True, 'do_crf':False, 'initln':False, 'initln_mean':0., 'initln_std':0.02, 'sample_weights':False}, (AdamW, {'correct_bias':False}, 'linwarm')],
    }
    MODEL_CONFIG_TEMPLATE_DEFAULTS = MODEL_CONFIG_TEMPLATE_VALUES['transformer-base']
    PREDEFINED_MODEL_CONFIG_KEYS = ['template', 'lm_mdl_name', 'lm_model', 'lm_params', 'clf', 'config', 'tknzr', 'lm_tknz_extra_char', 'kwargs']
    PREDEFINED_MODEL_CONFIG_VALUES = {
        'bert': ['transformer-base', 'bert-base-uncased', BertModel, 'BERT', M.BERTClfHead, BertConfig, BertTokenizer, ['[CLS]', '[SEP]', '[SEP]', '[MASK]'], {}],
    }
    # Common parameters
    TEMPLATE_VALUES_TYPE_MAP = {'task':TASK_CONFIG_TEMPLATE_VALUES, 'model':MODEL_CONFIG_TEMPLATE_VALUES}

    def __init__(self, dataset, model, template='', **kwargs):
        self.dataset = dataset
        self.model = model
        # Instantiation the parameters from template to the properties
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES:
            task_template = Configurable.PREDEFINED_TASK_CONFIG_VALUES[dataset][0]
            if task_template in Configurable.TASK_CONFIG_TEMPLATE_VALUES:
                self.__dict__.update(dict(zip(Configurable.TASK_CONFIG_TEMPLATE_KEYS, Configurable.TASK_CONFIG_TEMPLATE_VALUES[task_template])))
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES:
            model_template = Configurable.PREDEFINED_MODEL_CONFIG_VALUES[model][0]
        if model_template in Configurable.MODEL_CONFIG_TEMPLATE_VALUES:
            self.__dict__.update(dict(zip(Configurable.MODEL_CONFIG_TEMPLATE_KEYS, Configurable.MODEL_CONFIG_TEMPLATE_VALUES[model_template])))
        # Config some non-template attributes or overcome the replace the template values from the predefined parameters
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES:
            self.__dict__.update(dict(zip(Configurable.PREDEFINED_TASK_CONFIG_KEYS[1:-1], Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][1:-1])))
            self.__dict__.update(Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][-1])
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES:
            self.__dict__.update(dict(zip(Configurable.PREDEFINED_MODEL_CONFIG_KEYS[1:-1], Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][1:-1])))
            self.__dict__.update(Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][-1])

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _update_template_values(type, template_name, values={}):
        template_values = copy.deepcopy(Configurable.TEMPLATE_VALUES_TYPE_MAP[type][template_name])
        key_map = dict(zip(Configurable.TASK_CONFIG_TEMPLATE_KEYS, range(len(Configurable.TASK_CONFIG_TEMPLATE_KEYS))))
        for k, v in values.items():
            template_values[key_map[k]] = v
        return template_values

    def _get_template_value(self, name, idx):
        try:
        	return Configurable.TASK_CONFIG_TEMPLATE_VALUES[Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][0]][idx] if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES else self.__dict__.setdefault(name, Configurable.TASK_CONFIG_TEMPLATE_DEFAULTS[idx])
        except ValueError as e:
        	return Configurable.MODEL_CONFIG_TEMPLATE_VALUES[Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][0]][idx] if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES else self.__dict__.setdefault(name, Configurable.MODEL_CONFIG_TEMPLATE_DEFAULTS[idx])


    def __getattr__(self, name):
        if name in self.__dict__: return self.__dict__[name]
        try:
            try:
                attr_idx = Configurable.TASK_CONFIG_TEMPLATE_KEYS.index(name)
            except ValueError as e:
                attr_idx = Configurable.MODEL_CONFIG_TEMPLATE_KEYS.index(name)
            return self.__dict__.setdefault(name, self._get_template_value(name, attr_idx))
        except ValueError as e:
            return self.__dict__.setdefault(name, self.__dict__['_'+name] if '_'+name in self.__dict__ else None)


### Unit Test ###
def test_config():
    config = Configurable('biolarkgsc', 'bert')
    print(config.__dict__)
