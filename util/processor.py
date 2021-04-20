import os, sys, pickle, itertools

import numpy as np

import torch

import ftfy, spacy
try:
    nlp = spacy.load('en_core_sci_md')
except Exception as e:
    print(e)
    try:
        nlp = spacy.load('en_core_sci_sm')
    except Exception as e:
        print(e)
        nlp = spacy.load('en_core_web_sm')


def _sentclf_transform(sample, options=None, model=None, seqlen=32, start_tknids=[], clf_tknids=[], **kwargs):
    X, y = sample
    if model == 'bert' and (kwargs.setdefault('sentsim_func', None) is None or kwargs['sentsim_func']=='concat'):
        X = [start_tknids + x for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else start_tknids + X
    else: # GPT
        X = [start_tknids + x + clf_tknids for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else start_tknids + X + clf_tknids
    return X, y


def _entlmnt_transform(sample, options=None, model=None, seqlen=32, start_tknids=[], clf_tknids=[], delim_tknids=[], **kwargs):
    X, y = sample
    if model == 'bert':
        if kwargs.setdefault('sentsim_func', None) is None:
            trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], delim_tknids]]) - seqlen) / 2.0))
            X = [x[:len(x)-trim_len] for x in X]
            X = start_tknids + X[0] + delim_tknids + X[1] + delim_tknids
        else:
            pass
    else: # GPT
        if kwargs.setdefault('sentsim_func', None) is None:
            trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], clf_tknids]]) - seqlen) / 2.0))
            X = [x[:len(x)-trim_len] for x in X]
            X = start_tknids + X[0] + delim_tknids + X[1] + clf_tknids
        else:
            pass
    return X, y


def _sentsim_transform(sample, options=None, model=None, seqlen=32, start_tknids=[], clf_tknids=[], delim_tknids=[], **kwargs):
    X, y = sample
    if model == 'bert':
        if kwargs.setdefault('sentsim_func', None) is None:
            trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], delim_tknids]]) - seqlen) / 2.0))
            X = [x[:len(x)-trim_len] for x in X]
            X = start_tknids + X[0] + delim_tknids + X[1] + delim_tknids
        else:
            pass
    else: # GPT
        trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], clf_tknids]]) - seqlen) / 2.0))
        X = [x[:len(x)-trim_len] for x in X]
        X = [start_tknids + X[0] + delim_tknids + X[1] + clf_tknids, start_tknids + X[1] + delim_tknids + X[0] + clf_tknids]
    return X, y


def _padtrim_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None, **kwargs):
    X, y = sample
    X = [x[:min(seqlen, len(x))] + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))] + [xpad_val] * (seqlen - len(X))
    num_trim_delta = len([1 for x in X if seqlen > len(x)]) if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else 1 if seqlen > len(X) else 0
    if ypad_val is not None: y = [x[:min(seqlen, len(x))] + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))] + [ypad_val] * (seqlen - len(y))
    return X, y


def _trim_transform(sample, options=None, seqlen=32, trimlbs=False, required_special_tkns=[], special_tkns={}, **kwargs):
    seqlen -= sum([len(v) for v in special_tkns.values()])
    X, y = sample
    X = [x[:min(seqlen, len(x))] for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))]
    if trimlbs: y = [x[:min(seqlen, len(x))] for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))]
    return X, y


def _pad_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None, **kwargs):
    X, y = sample
    X = [x + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X + [xpad_val] * (seqlen - len(X))
    if ypad_val is not None: y = [x + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y + [ypad_val] * (seqlen - len(y))
    return X, y


def _dummy_trsfm(sample, **kwargs):
    return sample


def _adjust_encoder(mdl_name, tokenizer, config, extra_tokens=[], ret_list=False):
    encoded_extknids = []
    if (mdl_name.startswith('bert')):
        for tkn in extra_tokens:
            tkn_ids = tokenizer.tokenize(tkn)
            encoded_extknids.append([tokenizer.convert_tokens_to_ids(tkn_ids)] if (ret_list and type(tkn_ids) is not list) else tokenizer.convert_tokens_to_ids(tkn_ids))
    elif (mdl_name == 'gpt'):
        for tkn in extra_tokens:
            encoded_extknids.append([tokenizer.convert_tokens_to_ids(tkn)] if ret_list else tokenizer.convert_tokens_to_ids(tkn))
    elif (mdl_name == 'gpt2'):
        encoded_extknids = []
        for tkn in extra_tokens:
            tkn_ids = tokenizer.encode(tkn)
            encoded_extknids.append([tkn_ids] if (ret_list and type(tkn_ids) is not list) else tkn_ids)
    elif (mdl_name == 'trsfmxl'):
        for tkn in extra_tokens:
            tokenizer.__dict__[tkn] = len(tokenizer.__dict__)
            encoded_extknids.append([tokenizer.__dict__[tkn]] if ret_list else tokenizer.__dict__[tkn])
    elif (hasattr(config, 'embed_type') and mdl_name in config.embed_type):
        encoded_extknids = [[tkn] if ret_list else tkn for tkn in extra_tokens]
    else:
        encoded_extknids = [None] * len(extra_tokens)
    return encoded_extknids


def _base_encode(text, tokenizer):
    texts, records = [str(text)] if (type(text) is str or not hasattr(text, '__iter__')) else [str(s) for s in text], []
    try:
        for txt in texts:
            tokens = tokenizer.tokenize(ftfy.fix_text(txt))
            record = []
            while (len(tokens) > 512):
               record.extend(tokenizer.convert_tokens_to_ids(tokens[:512]))
               tokens = tokens[512:]
            record.extend(tokenizer.convert_tokens_to_ids(tokens))
            records.append(record)
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text).encode('ascii', 'replace').decode('ascii'))
        return []
    return records[0] if (type(text) is str or not hasattr(text, '__iter__')) else records


def _gpt2_encode(text, tokenizer):
    try:
        records = tokenizer.encode(ftfy.fix_text(str(text)).encode('ascii', 'replace').decode('ascii')) if (type(text) is str or not hasattr(text, '__iter__')) else [tokenizer.encode(ftfy.fix_text(str(line)).encode('ascii', 'replace').decode('ascii')) for line in text]
    except ValueError as e:
        try:
            records = list(itertools.chain(*[tokenizer.encode(w.text) for w in nlp(ftfy.fix_text(str(text)))])) if (type(text) is str or not hasattr(text, '__iter__')) else list(itertools.chain(*[list(itertools.chain(*[tokenizer.encode(w.text) for w in nlp(ftfy.fix_text(str(line)))])) for line in text]))
        except Exception as e:
            print(e)
            print('Cannot encode %s' % str(text.encode('ascii', 'replace').decode('ascii')))
            return []
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text.encode('ascii', 'replace').decode('ascii')))
        return []
    return records


def _batch2ids_w2v(batch_text, w2v_model):
    return [[w2v_model.vocab[w].index if w in w2v_model.vocab else (w2v_model.vocab[w.lower()].index if w.lower() in w2v_model.vocab else 0) for w in line] for line in batch_text]


def _batch2ids_sentvec(batch_text, sentvec_model):
    return torch.tensor(sentvec_model.embed_sentences([' '.join(x) for x in batch_text]))


def _onehot(y, size):
    y = torch.LongTensor(y).view(-1, 1)
    y_onehot = torch.FloatTensor(size[0], size[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot.long()
