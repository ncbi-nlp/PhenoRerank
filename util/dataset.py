import os, sys, random, operator, pickle, itertools
from collections import OrderedDict

import numpy as np
import pandas as pd

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.nn.parallel import replicate

from . import processor as P

SC=';;'


class DatasetInterface(object):
    def __init__(self):
        raise NotImplementedError

    def _transform_chain(self, sample):
        if self.transforms:
            self.transforms = self.transforms if type(self.transforms) is list else [self.transforms]
            self.transforms_kwargs = self.transforms_kwargs if type(self.transforms_kwargs) is list else [self.transforms_kwargs]
            for transform, transform_kwargs in zip(self.transforms, self.transforms_kwargs):
                transform_kwargs.update(self.transforms_args)
                sample = transform(sample, **transform_kwargs) if callable(transform) else getattr(self, transform)(sample, **transform_kwargs)
        return sample

    def _nmt_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], [self.binlb[y] if y in self.binlb else len(self.binlb) - 1 for y in sample[1]]

    def _mltl_nmt_transform(self, sample, options=None, binlb={}, get_lb=lambda x: x.split(SC)):
        if (len(binlb) > 0): self.binlb = binlb
        labels = [get_lb(lb) for lb in sample[1]]
        return sample[0], [[self.binlb[y] if y in self.binlb else len(self.binlb) - 1 for y in lbs] if type(lbs) is list else self.binlb[lbs] if lbs in self.binlb else len(self.binlb) - 1 for lbs in labels]

    def _binc_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], 1 if sample[1] in self.binlb else 0

    def _mltc_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], self.binlb.setdefault(sample[1], len(self.binlb))

    def _mltl_transform(self, sample, options=None, binlb={}, get_lb=lambda x: x.split(SC)):
        if (len(binlb) > 0): self.binlb = binlb
        labels = get_lb(sample[1])
        return sample[0], [1 if lb in labels else 0 for lb in self.binlb.keys()]

    def fill_labels(self, lbs, binlb=True, index=None, saved_col='preds', saved_path=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [(';'.join([self.binlbr[l] for l in np.where(lb == 1)[0]]) if self.mltl else ','.join(['_'.join([str(i), str(l)]) for i, l in enumerate(lb)])) if hasattr(lb, '__iter__') else (self.binlbr[lb] if len(self.binlbr) > 1 else (next(x for x in self.binlbr.values()) if lb==1 else '')) for lb in lbs]
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        try:
            if index is not None:
                filled_df.loc[index, saved_col] = lbs
            else:
                filled_df[saved_col] = lbs
        except Exception as e:
            print(e)
            with open('pred_lbs.tmp', 'wb') as fd:
                pickle.dump((filled_df, index, self.label_col, lbs), fd)
            raise e
        if (saved_path is not None):
            filled_df.to_csv(saved_path, sep='\t', **kwargs)
        return filled_df

    def rebalance(self):
        if (self.binlb is None): return
        task_cols, task_trsfm, task_extparms = self.config.task_col, self.config.task_trsfm, self.config.task_ext_trsfm
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        if (type(lb_df.iloc[0]) is list):
            lb_df[:] = [self._mltl_transform((None, SC.join(lbs)))[1] for lbs in lb_df]
            max_lb_df = lb_df.loc[[idx for idx, lbs in lb_df.iteritems() if np.sum(list(map(int, lbs))) == 0]]
            max_num, avg_num = max_lb_df.shape[0], 1.0 * lb_df[~lb_df.index.isin(max_lb_df.index)].shape[0] / len(lb_df.iloc[0])
        else:
            class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y is not None]).sum(axis=0)
            max_num, max_lb_bin = class_count.max(), class_count.argmax()
            max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
            avg_num = np.mean([class_count[x] for x in range(len(class_count)) if x != max_lb_bin])
        removed_idx = max_lb_df.sample(n=int(max_num-avg_num), random_state=1).index
        self.df = self.df.loc[list(set(self.df.index)-set(removed_idx))]

    def remove_mostfrqlb(self):
        if (self.binlb is None or self.binlb == 'rgrsn'): return
        task_cols, task_trsfm, task_extparms = self.config.task_col, self.config.task_trsfm, self.config.task_ext_trsfm
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y]).sum(axis=0)
        max_num, max_lb_bin = class_count.max(), class_count.argmax()
        max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
        self.df = self.df.loc[list(set(self.df.index)-set(max_lb_df.index))]


class BaseDataset(DatasetInterface, Dataset):
    """Basic dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, config, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], mltl=False, sampw=False, sampfrac=None, **kwargs):
        self.config = config
        self.text_col = [str(s) for s in text_col] if hasattr(text_col, '__iter__') and type(text_col) is not str else str(text_col)
        self.label_col = [str(s) for s in label_col] if hasattr(label_col, '__iter__') and type(label_col) is not str else str(label_col)
        self.df = self._df = csv_file if type(csv_file) is pd.DataFrame else pd.read_csv(csv_file, sep=sep, engine='python', error_bad_lines=False, dtype={self.label_col:'float' if binlb == 'rgrsn' else str}, **kwargs)
        print('Input DataFrame size: %s' % str(self.df.shape))
        if sampfrac: self.df = self._df = self._df.sample(frac=float(sampfrac))
        self.df.columns = self.df.columns.astype(str, copy=False)
        self.mltl = mltl
        self.sample_weights = sampw
        if (binlb == 'rgrsn'):
            self.binlb = None
            self.binlbr = None
            self.df = self.df[self.df[self.label_col].notnull()]
        elif (type(binlb) is str and binlb.startswith('mltl')):
            sc = binlb.split(SC)[-1]
            self.df = self.df[self.df[self.label_col].notnull()]
            lb_df = self.df[self.label_col]
            labels = sorted(set([lb for lbs in lb_df for lb in lbs.split(sc)])) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
            self.mltl = True
        elif (binlb is None):
            lb_df = self.df[self.df[self.label_col].notnull()][self.label_col]
            labels = sorted(set(lb_df)) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            if len(labels) == 1: labels = ['false'] + labels
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
        else:
            self.binlb = binlb
        if self.binlb: self.binlbr = OrderedDict([(i, lb) for lb, i in self.binlb.items()])
        self.encode_func = encode_func
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'vocab'):
            self.vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        self.transforms = transforms
        self.transforms_args = transforms_args
        self.transforms_kwargs = transforms_kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(record[self.text_col], self.tokenizer), record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], (sample[0] if type(sample[0]) is str or type(sample[0][0]) is str else torch.tensor(sample[0])), torch.tensor(sample[1])) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class SentSimDataset(BaseDataset):
    """Sentence Similarity task dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, config, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sampfrac=None, ynormfunc=None, **kwargs):
        super(SentSimDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, skip_blank_lines=False, keep_default_na=False, na_values=[], binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        self.ynormfunc, self.ynormfuncr = ynormfunc if ynormfunc is not None else (None, None)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], (sample[0] if type(sample[0][0]) is str or (type(sample[0][0]) is list and type(sample[0][0][0]) is str) else torch.tensor(sample[0])), torch.tensor(0. if sample[1] is np.nan else (float(sample[1]) if self.ynormfunc is None else self.ynormfunc(float(sample[1]))))) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        lbs = lbs if self.ynormfuncr is None else list(map(self.ynormfuncr, lbs))
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        if index:
            filled_df.loc[index, self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, **kwargs)
        return filled_df


class EntlmntDataset(BaseDataset):
    """Entailment task dataset class"""

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col] if self.label_col in record and record[self.label_col] is not np.nan else [k for k in [0, '0', 'false', 'False', 'F'] if k in self.binlb][0]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], (sample[0] if type(sample[0][0]) is str or (type(sample[0][0]) is list and type(sample[0][0][0]) is str) else torch.tensor(sample[0])), torch.tensor(sample[1])) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class MaskedLMDataset(BaseDataset):
    """Wrapper dataset class for masked language model"""

    def __init__(self, dataset, special_tknids=[101, 102, 102, 103], masked_lm_prob=0.15):
        self._ds = dataset
        self.config = self._ds.config
        self.text_col = self._ds.text_col if hasattr(self._ds, 'text_col') else None
        self.label_col = self._ds.label_col if hasattr(self._ds, 'label_col') else None
        self.mltl = self._ds.mltl if hasattr(self._ds, 'mltl') else None
        self.binlb = self._ds.binlb if hasattr(self._ds, 'binlb') else None
        self.binlbr = self._ds.binlbr if hasattr(self._ds, 'binlbr') else None
        self.encode_func = self._ds.encode_func
        self.tokenizer = self._ds.tokenizer
        self.vocab_size = self._ds.vocab_size
        self.transforms = self._ds.transforms
        self.transforms_args = self._ds.transforms_args
        self.transforms_kwargs = self._ds.transforms_kwargs
        self.special_tknids = special_tknids
        self.masked_lm_prob = masked_lm_prob
        if hasattr(self._ds, 'df'): self.df = self._ds.df
        if hasattr(self._ds, '_df'): self._df = self._ds._df

    def __len__(self):
        return self._ds.__len__()

    def __getitem__(self, idx):
        orig_sample = self._ds[idx]
        sample = orig_sample[1], orig_sample[2]
        masked_lm_ids = np.array(sample[0])
        pad_trnsfm_idx = self.transforms.index(P._pad_transform) if len(self.transforms) > 0 and P._pad_transform in self.transforms else -1
        pad_trnsfm_kwargs = self.transforms_kwargs[pad_trnsfm_idx] if pad_trnsfm_idx and pad_trnsfm_idx in self.transforms_kwargs > -1 else {}
        if type(self._ds) in [EntlmntDataset, SentSimDataset] and len(self.transforms_kwargs) >= 2 and self.transforms_kwargs[1].setdefault('sentsim_func', None) is not None:
            masked_lm_lbs = [np.array([-1 if x in self.special_tknids + [pad_trnsfm_kwargs.setdefault('xpad_val', -1)] else x for x in sample[0][X]]) for X in [0,1]]
            valid_idx = [np.where(masked_lm_lbs[x] > -1)[0] for x in [0,1]]
            cand_samp_idx = [random.sample(range(len(valid_idx[x])), min(self.config.maxlen, max(1, int(round(len(valid_idx[x]) * self.masked_lm_prob))))) for x in [0,1]]
            cand_idx = [valid_idx[x][cand_samp_idx[x]] for x in [0,1]]
            rndm = [np.random.uniform(low=0, high=1, size=(len(cand_idx[x]),)) for x in [0,1]]
            for x in [0,1]:
                masked_lm_ids[x][cand_idx[x][rndm[x] < 0.8]] = self.special_tknids[-1]
                masked_lm_ids[x][cand_idx[x][rndm[x] >= 0.9]] = random.randrange(0, self.vocab_size)
            for X in [0,1]:
                masked_lm_lbs[X][list(filter(lambda x: x not in cand_idx[X], range(len(masked_lm_lbs[X]))))] = -1
        else:
            masked_lm_lbs = np.array([-1 if x in self.special_tknids + [pad_trnsfm_kwargs.setdefault('xpad_val', -1)] else x for x in sample[0]])
            valid_idx = np.where(masked_lm_lbs > -1)[0]
            cand_samp_idx = random.sample(range(len(valid_idx)), min(self.config.maxlen, max(1, int(round(len(valid_idx) * self.masked_lm_prob)))))
            cand_idx = valid_idx[cand_samp_idx]
            rndm = np.random.uniform(low=0, high=1, size=(len(cand_idx),))
            masked_lm_ids[cand_idx[rndm < 0.8]] = self.special_tknids[-1]
            masked_lm_ids[cand_idx[rndm >= 0.9]] = random.randrange(0, self.vocab_size)
            masked_lm_lbs[list(filter(lambda x: x not in cand_idx, range(len(masked_lm_lbs))))] = -1
        segment_ids = torch.zeros(masked_lm_ids.shape)
        if type(self._ds) in [EntlmntDataset, SentSimDataset] and (len(self.transforms_kwargs) < 2 or self.transforms_kwargs[1].setdefault('sentsim_func', None) is None):
            segment_idx = sample[0].eq(self.special_tknids[2] * torch.ones_like(sample[0])).int()
            segment_idx = torch.where(segment_idx)
            segment_start, segment_end = torch.tensor([segment_idx[-1][i] for i in range(0, len(segment_idx[-1]), 2)]), torch.tensor([segment_idx[-1][i] for i in range(1, len(segment_idx[-1]), 2)])
            idx = torch.arange(sample[0].size(-1)) * torch.ones_like(sample[0])
            segment_ids = ((idx > segment_start.view(-1 if len(idx.size()) == 1 else (idx.size()[0], 1)) * torch.ones_like(idx)) & (idx <= segment_end.view(-1 if len(idx.size()) == 1 else (idx.size()[0], 1)) * torch.ones_like(idx))).int()
        return orig_sample + (torch.tensor(masked_lm_ids), torch.tensor(masked_lm_lbs), torch.tensor(segment_ids).long())

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        return self._ds.fill_labels(lbs, index=index, saved_path=saved_path, **kwargs)


class OntoDataset(BaseDataset):
    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, config, onto_fpath='onto.csv', onto_col='ontoid', sep='\t', index_col='id', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sampfrac=None, **kwargs):
        super(OntoDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, index_col=index_col, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        if hasattr(config, 'onto_df') and type(config.onto_df) is pd.DataFrame:
            self.onto = config.onto_df
        else:
            onto_fpath = config.onto if hasattr(config, 'onto') and os.path.exists(config.onto) else 'onto.csv'
            print('Reading ontology dictionary file [%s]...' % onto_fpath)
            self.onto = pd.read_csv(onto_fpath, sep=sep, index_col=index_col)
            setattr(config, 'onto_df', self.onto)
        print('Ontology DataFrame size: %s' % str(self.onto.shape))
        self.onto2id = dict([(k, i+1) for i, k in enumerate(self.onto.index)])
        self.onto_col = onto_col

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col] if self.label_col in record and record[self.label_col] is not np.nan else [k for k in [0, '0', 'false', 'False', 'F'] if k in self.binlb][0]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], (sample[0] if type(sample[0][0]) is str or (type(sample[0][0]) is list and type(sample[0][0][0]) is str) else torch.tensor(sample[0])), torch.tensor(sample[1])) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ()) + (torch.tensor(self.onto2id[record[self.onto_col]]),)


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def replicate(self, module, device_ids):
        replicas = super().replicate(module, device_ids)
        for attr in dir(module):
            attr_obj = getattr(module, attr)
            if type(attr_obj) is list and all([isinstance(x, nn.Module) for x in attr_obj]):
                rplcs = zip(*[replicate(x, device_ids, not torch.is_grad_enabled()) for x in attr_obj])
                for mdl, rplc in zip(replicas, rplcs):
                    setattr(mdl, attr, rplc)
            elif isinstance(attr_obj, nn.Module):
                for sub_attr in dir(attr_obj):
                    sub_attr_obj = getattr(attr_obj, sub_attr)
                    if type(sub_attr_obj) is list and all([isinstance(x, nn.Module) for x in sub_attr_obj]):
                        rplcs = zip(*[replicate(x, device_ids, not torch.is_grad_enabled()) for x in sub_attr_obj])
                        for mdl, rplc in zip(replicas, rplcs):
                            setattr(getattr(mdl, attr), sub_attr, rplc)
        return replicas
