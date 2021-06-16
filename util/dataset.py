import os, sys, random, operator, pickle, itertools, logging
from collections import OrderedDict

import numpy as np
import pandas as pd

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.nn.parallel import replicate

from . import processor as P

global FALSE_LABELS
FALSE_LABELS = [0, '0', 'false', 'False', 'F']


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

    def _nmt_transform(self, sample):
        return sample[0], [self.binlb.setdefault(y, len(self.binlb)) for y in sample[1]]

    def _mltl_nmt_transform(self, sample, get_lb=None):
        get_lb = (lambda x: x.split(self.sc)) if get_lb is None else get_lb
        labels = [get_lb(lb) for lb in sample[1]]
        return sample[0], [[self.binlb.setdefault(y, len(self.binlb)) for y in lbs] if type(lbs) is list else self.binlb.setdefault(lbs, len(self.binlb)) for lbs in labels]

    def _binc_transform(self, sample):
        return sample[0], 1 if sample[1] in self.binlb else 0

    def _mltc_transform(self, sample):
        return sample[0], self.binlb.setdefault(sample[1], len(self.binlb))

    def _mltl_transform(self, sample, get_lb=None):
        get_lb = (lambda x: x.split(self.sc)) if get_lb is None else get_lb
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
            logging.warning(e)
            with open('pred_lbs.tmp', 'wb') as fd:
                pickle.dump((filled_df, index, self.label_col, lbs), fd)
            raise e
        if (saved_path is not None):
            filled_df.to_csv(saved_path, sep='\t', **kwargs)
        return filled_df

    def rebalance(self):
        if (self.binlb is None): return
        task_cols, task_trsfm = self.config.task_col, self.config.task_trsfm
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        if (type(lb_df.iloc[0]) is list):
            lb_df[:] = [self._mltl_transform((None, self.sc.join(lbs)))[1] for lbs in lb_df]
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
        task_cols, task_trsfm = self.config.task_col, self.config.task_trsfm
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

    def __init__(self, csv_file, tokenizer, config, sampw=False, sampfrac=None, **kwargs):
        self.sc = config.sc
        self.config = config
        self.text_col = [str(s) for s in config.task_col['X']] if hasattr(config.task_col['X'], '__iter__') and type(config.task_col['X']) is not str else str(config.task_col['X'])
        self.label_col = [str(s) for s in config.task_col['y']] if hasattr(config.task_col['y'], '__iter__') and type(config.task_col['y']) is not str else str(config.task_col['y'])
        self.df = self._df = csv_file if type(csv_file) is pd.DataFrame else pd.read_csv(csv_file, sep=config.dfsep, engine='python', error_bad_lines=False, index_col=config.task_col['index'])
        self.df[config.task_col['y']] = ['false' if lb is None or lb is np.nan or (type(lb) is str and lb.isspace()) else lb for lb in self.df[config.task_col['y']]] # convert all absent labels to negative
        logging.info('Input DataFrame size: %s' % str(self.df.shape))
        if sampfrac: self.df = self._df = self._df.sample(frac=float(sampfrac))
        self.df.columns = self.df.columns.astype(str, copy=False)
        self.mltl = config.task_ext_params.setdefault('mltl', False)
        self.sample_weights = sampw

        # Construct the binary label mapping
        binlb = (config.task_ext_params['binlb'] if 'binlb' in config.task_ext_params else None) if 'binlb' not in kwargs else kwargs['binlb']
        if (binlb == 'rgrsn'): # regression tasks
            self.df[self.label_col] = self.df[self.label_col].astype('float')
            self.binlb = None
            self.binlbr = None
            self.df = self.df[self.df[self.label_col].notnull()]
        elif (type(binlb) is str and binlb.startswith('mltl')): # multi-label classification tasks
            sc = binlb.split(self.sc)[-1]
            self.df = self.df[self.df[self.label_col].notnull()]
            lb_df = self.df[self.label_col]
            labels = sorted(set([lb for lbs in lb_df for lb in lbs.split(sc)])) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
            self.mltl = True
        elif (binlb is None): # normal cases
            lb_df = self.df[self.df[self.label_col].notnull()][self.label_col]
            labels = sorted(set(lb_df)) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            if len(labels) == 1: labels = ['false'] + labels
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
        else: # previously constructed
            self.binlb = binlb
        if self.binlb: self.binlbr = OrderedDict([(i, lb) for lb, i in self.binlb.items()])
        self.encode_func = config.encode_func
        self.tknz_kwargs = config.tknz_kwargs
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'vocab'):
            self.vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        # Combine all the data transformers
        self.transforms = config.task_trsfm[0] + config.mdl_trsfm[0]
        self.transforms_kwargs = config.task_trsfm[1] + config.mdl_trsfm[1]
        self.transforms_args = kwargs.setdefault('transforms_args', {}) # Common transformer kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(record[self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]) if type(sample[0][0]) is not str else sample[0], torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]] if type(sample[0][0]) is not str else [torch.tensor(x) if x[0] is not str else x[0] for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

    @classmethod
    def callback_update_trsfm(cls, dataset):
        def _callback(config):
            dataset.transforms = config.task_trsfm[0] + config.mdl_trsfm[0]
            dataset.transforms_kwargs = config.task_trsfm[1] + config.mdl_trsfm[1]
        return _callback


class SentSimDataset(BaseDataset):
    """Sentence Similarity task dataset class"""

    def __init__(self, csv_file, tokenizer, config, sampw=False, sampfrac=None, **kwargs):
        super(SentSimDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, skip_blank_lines=False, keep_default_na=False, na_values=[], binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        self.ynormfunc, self.ynormfuncr = config.ynormfunc if hasattr(config, 'ynormfunc') else (None, None)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer, self.tknz_kwargs) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(0. if sample[1] is np.nan else (float(sample[1]) if self.ynormfunc is None else self.ynormfunc(float(sample[1]))))) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

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
        sample = self.encode_func([record[sent_idx] for sent_idx in self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col] if self.label_col in record and record[self.label_col] is not np.nan else [k for k in FALSE_LABELS if k in self.binlb][0]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class OntoDataset(BaseDataset):
    def __init__(self, csv_file, tokenizer, config, onto_col='ontoid', sampw=False, sampfrac=None, **kwargs):
        super(OntoDataset, self).__init__(csv_file, tokenizer, config, sampw=sampw, sampfrac=sampfrac, **kwargs)
        if hasattr(config, 'onto_df') and type(config.onto_df) is pd.DataFrame:
            self.onto = config.onto_df
        else:
            onto_fpath = config.onto if hasattr(config, 'onto') and os.path.exists(config.onto) else 'onto.csv'
            logging.info('Reading ontology dictionary file [%s]...' % onto_fpath)
            self.onto = pd.read_csv(onto_fpath, sep=config.dfsep, index_col=config.task_col['index'])
            setattr(config, 'onto_df', self.onto)
        logging.info('Ontology DataFrame size: %s' % str(self.onto.shape))
        self.onto2id = dict([(k, i+1) for i, k in enumerate(self.onto.index)])
        self.onto_col = onto_col

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func([record[sent_idx] for sent_idx in self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col] if self.label_col in record and record[self.label_col] is not np.nan else [k for k in [0, '0', 'false', 'False', 'F'] if k in self.binlb][0]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + (torch.tensor(self.onto2id[record[self.onto_col]]),) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


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
