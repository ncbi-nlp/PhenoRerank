import os, sys, ast, random, logging
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from transformers import get_linear_schedule_with_warmup

from bionlp.util import io

from util.config import *
from util.dataset import OntoDataset
from util.processor import _adjust_encoder
from util.trainer import train, eval
from util.common import _update_cfgs, gen_mdl, gen_clf, save_model, load_model

global FILE_DIR, CONFIG_FILE, DATA_PATH, SC, cfgr, args
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(FILE_DIR, 'etc', 'config.yaml')
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
SC=';;'
LB_SEP = ';'
args, cfgr = {}, None


def classify(dev_id=None):
    # config_kwargs = dict([(k, v) for k, v in args.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and not callable(v)])
    config = SimpleConfig.from_file_importmap(args.cfg.setdefault('config', 'config.json'), pkl_fpath=None, import_lib=True)
    # Prepare model related meta data
    mdl_name = config.model
    pr = io.param_reader(os.path.join(FILE_DIR, 'etc', '%s.yaml' % config.common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else {}
    _adjust_encoder(tokenizer, config)

    # Prepare task related meta data.
    task_path, task_type, task_dstype, task_cols, task_trsfm, task_extparms = config.input if config.input and os.path.isdir(os.path.join(DATA_PATH, config.input)) else config.task_path, config.task_type, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_params
    ds_kwargs = config.ds_kwargs

    # Prepare data
    if (not config.distrb or config.distrb and hvd.rank() == 0): logging.info('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train.%s' % config.fmt), tokenizer, config, **ds_kwargs)
    # Calculate the class weights if needed
    lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
    if (not config.weight_class or task_type == 'sentsim'):
        class_count = None
    elif len(lb_trsfm) > 0:
        lb_df = train_ds.df[task_cols['y']].apply(lb_trsfm[0])
        class_count = np.array([[1 if lb in y else 0 for lb in train_ds.binlb.keys()] for y in lb_df]).sum(axis=0)
    else:
        lb_df = train_ds.df[task_cols['y']]
        binlb = task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb
        class_count = lb_df.value_counts()[binlb.keys()].values
    if (class_count is None):
        class_weights = None
        sampler = None
    else:
        class_weights = torch.Tensor(1.0 / class_count)
        class_weights /= class_weights.sum()
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=config.bsize, replacement=True)
        if not config.distrb and type(dev_id) is list: class_weights = class_weights.repeat(len(dev_id))

    # Partition dataset among workers using DistributedSampler
    if config.distrb: sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(train_ds, batch_size=config.bsize, shuffle=sampler is None and config.droplast, sampler=sampler, num_workers=config.np, drop_last=config.droplast)

    # Classifier
    if (not config.distrb or config.distrb and hvd.rank() == 0):
        logging.info('Language model input fields: %s' % config.input_keys)
        logging.info('Classifier hyper-parameters: %s' % config.clf_ext_params)
        logging.info('Classifier task-related parameters: %s' % task_extparms['mdlaware'])
    if (config.resume):
        # Load model
        clf, prv_optimizer, resume, chckpnt = load_model(config.resume)
        if config.refresh:
            logging.info('Refreshing and saving the model with newest code...')
            try:
                if (not distrb or distrb and hvd.rank() == 0):
                    save_model(clf, prv_optimizer, '%s_%s.pth' % (config.task, config.model))
            except Exception as e:
                logging.warning(e)
        # Update parameters
        clf.update_params(task_params=task_extparms['mdlaware'], **config.clf_ext_params)
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=config.distrb)
        # Construct optimizer
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=config.lr, weight_decay=config.wdecay, **optmzr_cls[1]) if config.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=config.lr, momentum=0.9)
        if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())
        training_steps = int(len(train_ds) / config.bsize) if hasattr(train_ds, '__len__') else config.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(config.wrmprop*training_steps), num_training_steps=training_steps) if not config.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not config.distrb or config.distrb and hvd.rank() == 0): logging.info((optimizer, scheduler))
    else:
        # Build model
        lm_model, lm_config = gen_mdl(config, use_gpu=use_gpu, distrb=config.distrb, dev_id=dev_id)
        clf = gen_clf(config, lm_model, lm_config, num_lbs=len(train_ds.binlb) if train_ds.binlb else 1, mlt_trnsfmr=True if task_type in ['entlmnt', 'sentsim'] and task_extparms['mdlaware'].setdefault('sentsim_func', None) is not None else False, task_params=task_extparms['mdlaware'], binlb=train_ds.binlb, binlbr=train_ds.binlbr, use_gpu=use_gpu, distrb=config.distrb, dev_id=dev_id, **config.clf_ext_params)
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=config.lr, weight_decay=config.wdecay, **optmzr_cls[1]) if config.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=config.lr, momentum=0.9)
        training_steps = int(len(train_ds) / config.bsize) if hasattr(train_ds, '__len__') else config.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.wrmprop, num_training_steps=training_steps) if not config.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not config.distrb or config.distrb and hvd.rank() == 0): logging.info((optimizer, scheduler))

    if config.verbose: logging.debug(config.__dict__)

    if config.distrb:
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=clf.named_parameters())
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(clf.state_dict(), root_rank=0)

    # Training
    train(clf, optimizer, train_loader, config, scheduler, weights=class_weights, lmcoef=config.lmcoef, clipmaxn=config.clipmaxn, epochs=config.epochs, earlystop=config.earlystop, earlystop_delta=config.es_delta, earlystop_patience=config.es_patience, use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, resume=resume if config.resume else {})

    if config.distrb:
        if hvd.rank() == 0:
            clf = _handle_model(clf, dev_id=dev_id, distrb=False)
        else:
            return

    if config.noeval: return
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % config.fmt), tokenizer, config, binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, **ds_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=config.bsize, shuffle=False, num_workers=config.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % config.fmt), tokenizer, config, binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, **ds_kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.bsize, shuffle=False, num_workers=config.np)
    logging.debug(('binlb', train_ds.binlb, dev_ds.binlb, test_ds.binlb))

    # Evaluation
    eval(clf, dev_loader, config, ds_name='dev', use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
    if config.traindev: train(clf, optimizer, dev_loader, config, scheduler=scheduler, weights=class_weights, lmcoef=config.lmcoef, clipmaxn=config.clipmaxn, epochs=config.epochs, earlystop=config.earlystop, earlystop_delta=config.es_delta, earlystop_patience=config.es_patience, use_gpu=use_gpu, devq=dev_id, distrb=config.distrb)
    eval(clf, test_loader, config, ds_name='test', use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))


def _overlap_tuple(tuples, search, ret_idx=False):
    res = []
    for i, t in enumerate(tuples):
        if(t[1] > search[0] and t[0] < search[1]):
            res.append((i, t) if ret_idx else t)
    return (tuple(zip(*res)) if len(res) > 0 else ([],[])) if ret_idx else res


def split_df(df, lb_col='labels', lb_sep=';', keep_locs=True, update_locs=False):
    splitted_data = []
    for doc_id, row in df.iterrows():
        orig_labels = row[lb_col].split(lb_sep) if type(row[lb_col]) is str and row[lb_col] and not row[lb_col].isspace() else []
        if len(orig_labels) == 0:
            splitted_data.append(row.to_dict())
            continue
        labels, locs = zip(*[lb.split('|') for lb in orig_labels])
        doc = nlp(row['text'])
        sent_bndrs = [(s.start_char, s.end_char) for s in doc.sents]
        lb_locs = [tuple(map(int, loc.split(':'))) for loc in locs]
        if (np.amax(lb_locs) > np.amax(sent_bndrs)): lb_locs = np.array(lb_locs) - np.amin(lb_locs) # Temporary fix
        overlaps = list(filter(lambda x: len(x) > 0, [_overlap_tuple(lb_locs, sb, ret_idx=True) for sb in sent_bndrs]))
        indices, locss = zip(*overlaps) if len(overlaps) > 0 else ([[]]*len(sent_bndrs),[[]]*len(sent_bndrs))
        indices, locss = list(map(list, indices)), list(map(list, locss))
        miss_aligned = list(set(range(len(labels))) - set(itertools.chain.from_iterable(indices)))
        if len(miss_aligned) > 0:
            for i in range(len(sent_bndrs)):
                indices[i].extend(miss_aligned)
                locss[i].extend([sent_bndrs[i]]*len(miss_aligned))
        last_idx = 0
        for i, (idx, locs, sent) in enumerate(zip(indices, locss, doc.sents)):
            lbs = [labels[x] for x in idx]
            row['id'] = '_'.join(map(str, [doc_id, i]))
            row['text'] = sent.text
            row[lb_col] = lb_sep.join(['|'.join([lb, ':'.join(map(str, np.array(loc)-(last_idx if update_locs else 0)))]) for lb, loc in zip(lbs, locs)] if keep_locs else lbs)
            last_idx += sent.end_char
            splitted_data.append(row.to_dict())
    splitted_df = pd.DataFrame(splitted_data)
    return splitted_df


def mltl2entlmnt(in_fpath, onto_dict, out_fpath=None, lb_col='labels', pred_col='preds', sent_mode=False, max_num_samp=1):
    df = pd.read_csv(in_fpath, sep='\t', index_col='id')
    if sent_mode: df = split_df(df, lb_col=pred_col, lb_sep=LB_SEP, keep_locs=False).set_index('id')
    pid, text1, text2, ontos, label, label1, label2 = [[] for x in range(7)]
    for doc_id, row in df.iterrows():
        labels = row[lb_col].split(LB_SEP) if type(row[lb_col]) is str and row[lb_col] and not row[lb_col].isspace() else []
        labels = list(map(lambda x: x.split('|')[0], labels))
        preds = row[pred_col].split(LB_SEP) if type(row[pred_col]) is str and row[pred_col] and not row[pred_col].isspace() else []
        preds = list(map(lambda x: x.split('|')[0], preds))
        names = [(onto_dict.loc[lb]['label'].tolist()[0] if type(onto_dict.loc[lb]['label']) is pd.Series else onto_dict.loc[lb]['label']) if lb in onto_dict.index else [] for lb in preds]
        notes = [(list(set(onto_dict.loc[lb]['text'].tolist())) if type(onto_dict.loc[lb]['text']) is pd.Series else [onto_dict.loc[lb]['text']]) if lb in onto_dict.index else [[t for t in onto_dict.loc[lb][['label','exact_synm','relate_synm','narrow_synm','broad_synm']] if t is not np.nan][0]] if lb in onto_dict.index else [] for lb in preds if lb]
        for i, lb, name, note in zip(range(len(preds)), preds, names, notes):
            if len(note) == 0: continue
            note = note[:min(len(note), max_num_samp)]
            for j, ntxt in enumerate(note):
                pid.append('%s_%i_%i' % (doc_id, i, j))
                text1.append(row['text'])
                text2.append(ntxt)
                label.append('include' if lb in labels else '')
            ontos.extend([name]*len(note))
            label1.extend([';'.join(labels)]*len(note))
            label2.extend([lb]*len(note))
    entlmnt_df = pd.DataFrame(OrderedDict(id=pid, text1=text1, text2=text2, onto=ontos, label=label, label1=label1, label2=label2))
    entlmnt_df.to_csv(('%s_entlmnt.csv' % os.path.splitext(in_fpath)[0]) if out_fpath is None else out_fpath, sep='\t', index=None, encoding='utf-8')

def seqmatch(text, query):
    import difflib
    s = difflib.SequenceMatcher(None, text, query)
    return sum(n for i,j,n in s.get_matching_blocks()) / float(len(query))

def match_dict(text, cid, dictionary, fields=['label'], metric=seqmatch):
    return np.max([metric(text, lb) for lb in dictionary.loc[cid][fields] if type(lb) is str])

NUM_UNDERLINE_IN_ORIG = {'biolarkgsc':0, 'copd':1}

def entlmnt2mltl(in_fpath, ref_df, onto_dict, out_fpath=None, idx_num_underline=0, text_sim=seqmatch, sim_thrshld=0.9):
    sms_pred_df = pd.read_csv(in_fpath, sep='\t', encoding='utf-8', engine='python')
    sms_pred_df['orig_id'] = ['_'.join(str(idx).split('_')[:idx_num_underline+1]) for idx in sms_pred_df['id']]
    merged_data = {'id':[], 'labels':[]}
    for gid, grp in sms_pred_df.groupby('orig_id'):
        merged_lbs = [lb for text, lb, pred in zip(grp['text1'], grp['label2'], grp['preds']) if pred == 'include' or match_dict(text, lb, onto_dict, fields=['label', 'exact_synm', 'relate_synm', 'narrow_synm', 'broad_synm'], metric=text_sim)>sim_thrshld]
        merged_data['id'].append(gid)
        merged_data['labels'].append(LB_SEP.join(merged_lbs) if len(merged_lbs) > 0 else '')
    pred_df = ref_df.copy()
    not_pred_ids = set(map(str, pred_df['id'].values)) - set(merged_data['id'])
    filtered_ids = [idx for idx in pred_df['id'] if idx not in not_pred_ids]
    pred_df['preds'] = [[]] * pred_df.shape[0]
    pred_df.loc[pred_df['id'].apply(lambda x: x not in not_pred_ids), 'preds'] = pd.DataFrame(merged_data).set_index('id').loc[filtered_ids]['labels'].tolist()
    pred_df.to_csv(('%s_reranked.csv' % os.path.splitext(in_fpath)[0]) if out_fpath is None else out_fpath, sep='\t', index=None, encoding='utf-8')


def rerank(dev_id=None):
    print('### Re-rank Mode ###')
    orig_task = args.task
    args.task = '%s_entilement' % orig_task
    # Prepare model related meta data
    mdl_name = args.model.split('_')[0].lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(FILE_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in args.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and not callable(v)])
    orig_config = Configurable(orig_task, mdl_name, common_cfg=common_cfg, wsdir=FILE_DIR, sc=SC, **config_kwargs)
    config = Configurable(args.task, mdl_name, common_cfg=common_cfg, wsdir=FILE_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = config.task_type
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',args.model), ('sentsim_func', args.sentsim_func), ('seqlen',args.maxlen)])
    # Prepare task related meta data
    args.input = '%s_entlmnt.csv' % orig_task
    onto_dict = pd.read_csv(args.onto if args.onto and os.path.exists(args.onto) else 'hpo_dict.csv', sep='\t', index_col='id', encoding='utf-8')
    mltl2entlmnt(args.prvres if args.prvres and os.path.exists(args.prvres) else os.path.join(DATA_PATH, orig_config.task_path, 'test.%s' % args.fmt), onto_dict, out_fpath=args.input, sent_mode=args.sent)
    task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':args.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':args.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':args.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':args.sample_weights, 'sampfrac':args.sampfrac}
    ds_kwargs.update(dict((k, task_extparms[k]) for k in ['origlb', 'locallb', 'lbtxt', 'neglbs', 'reflb', 'sent_mode'] if k in task_extparms))
    if task_dstype in [OntoDataset]:
        ds_kwargs['onto_fpath'] = args.onto if args.onto else task_extparms.setdefault('onto_fpath', 'onto.csv')
        ds_kwargs['onto_col'] = task_cols['ontoid']
    task_params = dict([(k, getattr(args. k)) if hasattr(args. k) and getattr(args. k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])

    # Load model
    clf, prv_optimizer, resume, chckpnt = load_model(args.resume)
    if args.refresh:
        print('Refreshing and saving the model with newest code...')
        try:
            save_model(clf, prv_optimizer, '%s_%s.pth' % (args.task, args.model))
        except Exception as e:
            print(e)
    prv_task_params = copy.deepcopy(clf.task_params)
    # Update parameters
    clf.update_params(task_params=task_params, sample_weights=False)
    if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=args.distrb)
    optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
    optimizer = optmzr_cls[0](clf.parameters(), lr=args.lr, weight_decay=args.wdecay, **optmzr_cls[1]) if args.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=args.lr, momentum=0.9)
    if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())

    # Prepare test set
    test_ds = task_dstype(args.input, task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, **ds_kwargs)
    if mdl_name == 'bert': test_ds = MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=args.bsize, shuffle=False, num_workers=args.np)

    # Evaluation
    if args.traindev:
        dev_ds = task_dstype(args.input, task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, **ds_kwargs)
        training_steps = int(len(v) / args.bsize) if hasattr(dev_ds, '__len__') else args.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.wrmprop*training_steps), num_training_steps=training_steps) if not args.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not args.distrb or args.distrb and hvd.rank() == 0): print((optimizer, scheduler))
        if mdl_name == 'bert': dev_ds = MaskedLMDataset(dev_ds)
        dev_loader = DataLoader(dev_ds, batch_size=args.bsize, shuffle=False, num_workers=args.np)
        eval(clf, dev_loader, config, dev_ds.binlbr, special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=args.task, ds_name='dev', mdl_name=args.model, use_gpu=use_gpu, devq=dev_id, distrb=args.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
        train(clf, optimizer, dev_loader, special_tknids_args, scheduler=scheduler, pad_val=task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=args.lmcoef, clipmaxn=args.clipmaxn, epochs=args.epochs, earlystop=args.earlystop, earlystop_delta=args.es_delta, earlystop_patience=args.es_patience, task_type=task_type, task_name=args.task, mdl_name=args.model, use_gpu=use_gpu, devq=dev_id, distrb=args.distrb)
    eval(clf, test_loader, config, test_ds.binlbr, special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=args.task, ds_name='test', mdl_name=args.model, use_gpu=use_gpu, devq=dev_id, distrb=args.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
    os.rename('pred_test.csv', '%s_entlmnt_pred.csv' % orig_task)
    ref_df = pd.read_csv(args.prvres if args.prvres and os.path.exists(args.prvres) else os.path.join(DATA_PATH, orig_config.task_path, 'test.%s' % args.fmt), sep='\t', dtype={'id':object}, encoding='utf-8')
    entlmnt2mltl('%s_entlmnt_pred.csv' % orig_task, ref_df, onto_dict, out_fpath='%s_rerank_pred_test.csv' % orig_task, idx_num_underline=NUM_UNDERLINE_IN_ORIG[orig_task])


def main():
    predefined_tasks = ['biolarkgsc', 'copd', 'biolarkgsc_entilement', 'copd_entilement', 'hpo_entilement']
    if any(args.task == t for t in ['biolarkgsc_entilement', 'copd_entilement', 'hpo_entilement']):
        if args.method != 'train':
            print('Running in training mode instead ...')
        main_func = classify
    elif any(args.task == t for t in ['biolarkgsc', 'copd']):
        if args.method != 'rerank':
            print('Running in re-ranking mode instead ...')
        main_func = rerank
    else:
        print('Please select the task among: %s' % predefined_tasks)
        return

    if (args.distrb or args.devq):
        main_func(args.devq if len(args.devq) > 1 else args.devq[0])
    else:
        main_func(None) # CPU


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Train or evaluate the re-ranking model.')
    parser.add_argument('-p', '--pid', default=0, action='store', type=int, dest='pid', help='indicate the process ID')
    parser.add_argument('-n', '--np', default=1, action='store', type=int, dest='np', help='indicate the number of processes used for training')
    parser.add_argument('-f', '--fmt', default='tsv', help='data stored format: tsv or csv [default: %default]')
    parser.add_argument('-j', '--epochs', default=1, action='store', type=int, dest='epochs', help='indicate the epoch used in deep learning')
    parser.add_argument('-z', '--bsize', default=64, action='store', type=int, dest='bsize', help='indicate the batch size used in deep learning')
    parser.add_argument('-g', '--gpunum', default=1, action='store', type=int, dest='gpunum', help='indicate the gpu device number')
    parser.add_argument('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
    parser.add_argument('--gpumem', default=0.5, action='store', type=float, dest='gpumem', help='indicate the per process gpu memory fraction')
    parser.add_argument('--distrb', default=False, action='store_true', dest='distrb', help='whether to distribute data over multiple devices')
    parser.add_argument('--distbknd', default='nccl', action='store', dest='distbknd', help='distribute framework backend')
    parser.add_argument('--disturl', default='env://', action='store', dest='disturl', help='distribute framework url')
    parser.add_argument('--optim', default='adam', action='store', dest='optim', help='indicate the optimizer')
    parser.add_argument('--wrmprop', default=0.1, action='store', type=float, dest='wrmprop', help='indicate the warmup proportion')
    parser.add_argument('--trainsteps', default=1000, action='store', type=int, dest='trainsteps', help='indicate the training steps')
    parser.add_argument('--traindev', default=False, action='store_true', dest='traindev', help='whether to use dev dataset for training')
    parser.add_argument('--noeval', default=False, action='store_true', dest='noeval', help='whether to train only')
    parser.add_argument('--noschdlr', default=False, action='store_true', dest='noschdlr', help='force to not use scheduler whatever the default setting is')
    parser.add_argument('--earlystop', default=False, action='store_true', dest='earlystop', help='whether to use early stopping')
    parser.add_argument('--es_patience', default=5, action='store', type=int, dest='es_patience', help='indicate the tolerance time for training metric violation')
    parser.add_argument('--es_delta', default=float(5e-3), action='store', type=float, dest='es_delta', help='indicate the minimum delta of early stopping')
    parser.add_argument('--vocab', dest='vocab', help='vocabulary file')
    parser.add_argument('--bpe', dest='bpe', help='bpe merge file')
    parser.add_argument('--w2v', dest='w2v_path', help='word2vec model file')
    parser.add_argument('--sentvec', dest='sentvec_path', help='sentvec model file')
    parser.add_argument('--maxlen', default=128, action='store', type=int, dest='maxlen', help='indicate the maximum sequence length for each samples')
    parser.add_argument('--maxtrial', default=50, action='store', type=int, dest='maxtrial', help='maximum time to try')
    parser.add_argument('--initln', default=False, action='store_true', dest='initln', help='whether to initialize the linear layer')
    parser.add_argument('--initln_mean', default=0., action='store', type=float, dest='initln_mean', help='indicate the mean of the parameters in linear model when Initializing')
    parser.add_argument('--initln_std', default=0.02, action='store', type=float, dest='initln_std', help='indicate the standard deviation of the parameters in linear model when Initializing')
    parser.add_argument('--weight_class', default=False, action='store_true', dest='weight_class', help='whether to drop the last incompleted batch')
    parser.add_argument('--clswfac', default='1', action='store', type=str, dest='clswfac', help='whether to drop the last incompleted batch')
    parser.add_argument('--droplast', default=False, action='store_true', dest='droplast', help='whether to drop the last incompleted batch')
    parser.add_argument('--do_norm', default=False, action='store_true', dest='do_norm', help='whether to do normalization')
    parser.add_argument('--norm_type', default='batch', action='store', dest='norm_type', help='normalization layer class')
    parser.add_argument('--do_extlin', default=False, action='store_true', dest='do_extlin', help='whether to apply additional fully-connected layer to the hidden states of the language model')
    parser.add_argument('--do_lastdrop', default=False, action='store_true', dest='do_lastdrop', help='whether to apply dropout to the last layer')
    parser.add_argument('--lm_loss', default=False, action='store_true', dest='lm_loss', help='whether to apply dropout to the last layer')
    parser.add_argument('--do_crf', default=False, action='store_true', dest='do_crf', help='whether to apply CRF layer')
    parser.add_argument('--do_thrshld', default=False, action='store_true', dest='do_thrshld', help='whether to apply ThresholdEstimator layer')
    parser.add_argument('--fchdim', default=0, action='store', type=int, dest='fchdim', help='indicate the dimensions of the hidden layers in the Embedding-based classifier, 0 means using only one linear layer')
    parser.add_argument('--iactvtn', default='relu', action='store', dest='iactvtn', help='indicate the internal activation function')
    parser.add_argument('--oactvtn', default='sigmoid', action='store', dest='oactvtn', help='indicate the output activation function')
    parser.add_argument('--bert_outlayer', default='-1', action='store', type=str, dest='output_layer', help='indicate which layer to be the output of BERT model')
    parser.add_argument('--pooler', dest='pooler', help='indicate the pooling strategy when selecting features: max or avg')
    parser.add_argument('--seq2seq', dest='seq2seq', help='indicate the seq2seq strategy when converting sequences of embeddings into a vector')
    parser.add_argument('--seq2vec', dest='seq2vec', help='indicate the seq2vec strategy when converting sequences of embeddings into a vector: pytorch-lstm, cnn, or cnn_highway')
    parser.add_argument('--ssfunc', dest='sentsim_func', help='indicate the sentence similarity metric [dist|sim]')
    parser.add_argument('--catform', dest='concat_strategy', help='indicate the sentence similarity metric [normal|diff]')
    parser.add_argument('--ymode', default='sim', dest='ymode', help='indicate the sentence similarity metric in gold standard [dist|sim]')
    parser.add_argument('--loss', dest='loss', help='indicate the loss function')
    parser.add_argument('--cnstrnts', dest='cnstrnts', help='indicate the constraint scheme')
    parser.add_argument('--lr', default=float(1e-3), action='store', type=float, dest='lr', help='indicate the learning rate of the optimizer')
    parser.add_argument('--wdecay', default=float(1e-5), action='store', type=float, dest='wdecay', help='indicate the weight decay of the optimizer')
    parser.add_argument('--lmcoef', default=0.5, action='store', type=float, dest='lmcoef', help='indicate the coefficient of the language model loss when fine tuning')
    parser.add_argument('--sampfrac', action='store', type=float, dest='sampfrac', help='indicate the sampling fraction for datasets')
    parser.add_argument('--pdrop', default=0.2, action='store', type=float, dest='pdrop', help='indicate the dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler')
    parser.add_argument('--pthrshld', default=0.5, action='store', type=float, dest='pthrshld', help='indicate the threshold for predictive probabilitiy')
    parser.add_argument('--topk', default=5, action='store', type=int, dest='topk', help='indicate the top k search parameter')
    parser.add_argument('--do_tfidf', default=False, action='store_true', dest='do_tfidf', help='whether to use tfidf as text features')
    parser.add_argument('--do_chartfidf', default=False, action='store_true', dest='do_chartfidf', help='whether to use charater tfidf as text features')
    parser.add_argument('--do_bm25', default=False, action='store_true', dest='do_bm25', help='whether to use bm25 as text features')
    parser.add_argument('--clipmaxn', action='store', type=float, dest='clipmaxn', help='indicate the max norm of the gradients')
    parser.add_argument('--resume', action='store', dest='resume', help='resume training model file')
    parser.add_argument('--refresh', default=False, action='store_true', dest='refresh', help='refresh the trained model with newest code')
    parser.add_argument('--sampw', default=False, action='store_true', dest='sample_weights', help='use sample weights')
    parser.add_argument('--corpus', help='corpus data')
    parser.add_argument('--onto', help='ontology data')
    parser.add_argument('--pred', help='prediction file')
    parser.add_argument('--sent', default=False, action='store_true', dest='sent', help='whether to use location of labels to split the text into sentences')
    parser.add_argument('--datapath', help='location of dataset')
    parser.add_argument('-i', '--input', help='input dataset')
    parser.add_argument('--prvres', help='previous results for re-ranking')
    parser.add_argument('-w', '--cache', default='.cache', help='the location of cache files')
    parser.add_argument('-u', '--task', default='ddi', type=str, dest='task', help='the task name [default: %default]')
    parser.add_argument('--model', default='gpt2', type=str, dest='model', help='the model to be validated')
    parser.add_argument('--encoder', dest='encoder', help='the encoder to be used after the language model: pool, s2v or s2s')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model file')
    parser.add_argument('--seed', dest='seed', help='manually set the random seed')
    parser.add_argument('--dfsep', default='\t', help='separate character for pandas dataframe')
    parser.add_argument('--sc', default=';;', help='separate character for multiple-value records')
    parser.add_argument('-c', '--cfg', help='config string used to update the settings, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
    parser.add_argument('-m', '--method', default='rerank', help='main method to run')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', dest='verbose', help='display detailed information')
    args = parser.parse_args()

    # Logging setting
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Parse config file
    if (os.path.exists(CONFIG_FILE)):
    	cfgr = io.cfg_reader(CONFIG_FILE)
    else:
        logging.error('Config file `%s` does not exist!' % CONFIG_FILE)
        sys.exit(1)

    # Update config
    cfg_kwargs = {} if args.cfg is None else ast.literal_eval(args.cfg)
    args.cfg = cfg_kwargs
    _update_cfgs(globals(), cfg_kwargs)

    # GPU setting
    if (args.gpuq is not None and not args.gpuq.strip().isspace()):
    	args.gpuq = list(range(torch.cuda.device_count())) if (args.gpuq == 'auto' or args.gpuq == 'all') else [int(x) for x in args.gpuq.split(',') if x]
    elif (args.gpunum > 0):
        args.gpuq = list(range(args.gpunum))
    else:
        args.gpuq = []
    if (args.gpuq and args.gpunum > 0):
        if args.verbose: os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[:args.gpunum]) if 'CUDA_VISIBLE_DEVICES' in os.environ and not os.environ['CUDA_VISIBLE_DEVICES'].isspace() else ','.join(map(str, args.gpuq[:args.gpunum]))
        setattr(args, 'devq', list(range(torch.cuda.device_count())))
    else:
        setattr(args, 'devq', None)
    if args.distrb:
        import horovod.torch as hvd
        hvd.init()
        DATA_PATH = os.path.join('/', 'data', 'bionlp')
        torch.cuda.set_device(hvd.local_rank())

    # Process config
    if args.datapath is not None: DATA_PATH = args.datapath
    SC = args.sc

    # Random seed setting
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Format some arguments
    args.output_layer = list(map(int, args.output_layer.split(',')))
    args.output_layer = args.output_layer[0] if len(args.output_layer) == 1 else args.output_layer
    args.clswfac = list(map(float, args.clswfac.split(','))) if ',' in args.clswfac else float(args.clswfac)

    main()
