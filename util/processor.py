import os, sys, pickle, itertools, logging

import numpy as np

import torch

import ftfy, spacy
try:
    nlp = spacy.load('en_core_sci_md')
except Exception as e:
    logging.warning(e)
    try:
        nlp = spacy.load('en_core_sci_sm')
    except Exception as e:
        logging.warning(e)
        nlp = spacy.load('en_core_web_sm')


def _base_transform(sample, input_keys=[]):
    X, y = sample
    X = [X[k] for k in input_keys]
    if len(X) == 0: raise Exception('No inputs generated!')
    return X, y


def _bert_input_keys(task_type=None):
    return ['input_ids', 'attention_mask'] + ([] if task_type is None or task_type not in ['entlmnt', 'sentsim'] else ['token_type_ids'])

def _bert_onto_input_keys(task_type=None):
    return _bert_input_keys(task_type=None) + ['onto_id']


def _adjust_encoder(tokenizer, config):
    encoded_extknids = []
    if (config.model.startswith('bert')):
        pass
    elif (config.model == 'gpt'):
        tokenizer.cls_token, tokenizer.eos_token, tokenizer.pad_token = '<CLS>', '<EOS>', '<PAD>'
    elif (config.model == 'gpt2'):
        tokenizer.pad_token = tokenizer.eos_token
    elif (config.model == 'trsfmxl'):
        pass
    elif (hasattr(config, 'embed_type') and config.model in config.embed_type):
        tokenizer.update({'bos_token':'<BOS>', 'eos_token':'<EOS>', 'pad_token':'<PAD>'})
    else:
        pass


def _base_encode(text, tokenizer, tknz_kwargs={}):
    try:
        try:
            record = tokenizer(text, **tknz_kwargs) if (type(text) is str or not hasattr(text, '__iter__')) else tokenizer(*text, **tknz_kwargs)
        except Exception as e:
            record = tokenizer(ftfy.fix_text(text), **tknz_kwargs) if (type(text) is str or not hasattr(text, '__iter__')) else tokenizer(*[ftfy.fix_text(str(s)) for s in text], **tknz_kwargs)
        return record
    except Exception as e:
        logging.warning(e)
        logging.warning('Cannot encode %s' % str(text).encode('ascii', 'replace').decode('ascii'))
        return {}
