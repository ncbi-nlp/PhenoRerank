import os, sys, ast, random, logging
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from transformers import get_linear_schedule_with_warmup

from util.config import *
from util.dataset import OntoDataset
from util.processor import _adjust_encoder
from util.trainer import train, eval
from util.common import _update_cfgs, param_reader, write_json, read_json, gen_mdl, gen_clf, _handle_model, save_model, load_model, seqmatch, mltl2entlmnt, entlmnt2mltl

global FILE_DIR, DATA_PATH, args
FILE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
args = {}


def classify(dev_id=None):
    config_updates = dict([(k, v) for k, v in args.__dict__.items() if not k.startswith('_') and k not in set(['model', 'cfg']) and v is not None and not callable(v)])
    config_updates.update({**args.cfg, **{'wsdir':FILE_DIR}})
    config = SimpleConfig.from_file_importmap(args.cfg.setdefault('config', 'config.json'), pkl_fpath=None, import_lib=True, updates=config_updates)
    # Prepare model related meta data
    mdl_name = config.model
    pr = param_reader(os.path.join(FILE_DIR, 'etc', '%s.yaml' % config.common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
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
        class_weights *= (args.clswfac[min(len(args.clswfac)-1, i)] if type(args.clswfac) is list else args.clswfac)
        sampler = None # WeightedRandomSampler does not work in new version
        # sampler = WeightedRandomSampler(weights=class_weights, num_samples=config.bsize, replacement=True)
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
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

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


def rerank(dev_id=None):
    print('### Re-rank Mode ###')
    orig_task = args.task
    args.task = '%s_entilement' % orig_task
    config_updates = dict([(k, v) for k, v in args.__dict__.items() if not k.startswith('_') and k not in set(['model', 'cfg']) and v is not None and not callable(v)])
    config_updates.update({**args.cfg, **{'wsdir':FILE_DIR}})
    config = SimpleConfig.from_file_importmap(args.cfg.setdefault('config', 'config.json'), pkl_fpath=None, import_lib=True, updates=config_updates)
    # Prepare model related meta data
    mdl_name = config.model
    pr = param_reader(os.path.join(FILE_DIR, 'etc', '%s.yaml' % config.common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    # Prepare task related meta data.
    task_path, task_type, task_dstype, task_cols, task_trsfm, task_extparms = config.input if config.input and os.path.isdir(os.path.join(DATA_PATH, config.input)) else config.task_path, config.task_type, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_params
    ds_kwargs = config.ds_kwargs
    config.input = '%s_entlmnt.csv' % orig_task
    onto_dict = pd.read_csv(config.onto, sep='\t', index_col='id', encoding='utf-8')
    mltl2entlmnt(config.prvres if config.prvres and os.path.exists(config.prvres) else os.path.join(DATA_PATH, orig_config.task_path, 'test.%s' % config.fmt), onto_dict, out_fpath=config.input, sent_mode=config.sent)

    if config.verbose: logging.debug(config.__dict__)

    # Load model
    clf, prv_optimizer, resume, chckpnt = load_model(config.resume)
    if config.refresh:
        print('Refreshing and saving the model with newest code...')
        try:
            save_model(clf, prv_optimizer, '%s_%s.pth' % (config.task, config.model))
        except Exception as e:
            print(e)
    # Update parameters
    clf.update_params(task_params=task_extparms['mdlaware'], **config.clf_ext_params)
    if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=config.distrb)
    optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
    optimizer = optmzr_cls[0](clf.parameters(), lr=config.lr, weight_decay=config.wdecay, **optmzr_cls[1]) if config.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=config.lr, momentum=0.9)
    if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())

    # Prepare test set
    test_ds = task_dstype(config.input, tokenizer, config, ds_name='test', binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, **ds_kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.bsize, shuffle=False, num_workers=config.np)

    # Evaluation
    if config.traindev:
        dev_ds = task_dstype(config.input, tokenizer, config, ds_name='dev', binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, **ds_kwargs)
        training_steps = int(len(v) / config.bsize) if hasattr(dev_ds, '__len__') else config.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.wrmprop, num_training_steps=training_steps) if not config.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not config.distrb or config.distrb and hvd.rank() == 0): print((optimizer, scheduler))
        dev_loader = DataLoader(dev_ds, batch_size=config.bsize, shuffle=False, num_workers=config.np)
        eval(clf, dev_loader, config, ds_name='dev', use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
        train(clf, optimizer, dev_loader, config, scheduler, weights=class_weights, lmcoef=config.lmcoef, clipmaxn=config.clipmaxn, epochs=config.epochs, earlystop=config.earlystop, earlystop_delta=config.es_delta, earlystop_patience=config.es_patience, use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, resume=resume if config.resume else {})
    eval(clf, test_loader, config, ds_name='test', use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
    os.rename('pred_test.csv', '%s_entlmnt_pred.csv' % orig_task)
    ref_df = pd.read_csv(config.prvres if config.prvres and os.path.exists(config.prvres) else os.path.join(DATA_PATH, orig_config.task_path, 'test.%s' % config.fmt), sep='\t', dtype={'id':object}, encoding='utf-8')
    entlmnt2mltl('%s_entlmnt_pred.csv' % orig_task, ref_df, onto_dict, out_fpath='%s_rerank_pred_test.csv' % orig_task, idx_num_underline=NUM_UNDERLINE_IN_ORIG[orig_task])


def main():
    predefined_tasks = ['biolarkgsc', 'copd', 'biolarkgsc_entilement', 'copd_entilement', 'hpo_entilement']
    if any(args.task == t for t in ['biolarkgsc_entilement', 'copd_entilement', 'hpo_entilement']):
        if args.method != 'train' or args.method != 'fine-tune':
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
    parser.add_argument('-j', '--epochs', default=1, action='store', type=int, dest='epochs', help='indicate the epoch used in deep learning')
    parser.add_argument('-z', '--bsize', default=64, action='store', type=int, dest='bsize', help='indicate the batch size used in deep learning')
    parser.add_argument('-g', '--gpunum', default=1, action='store', type=int, dest='gpunum', help='indicate the gpu device number')
    parser.add_argument('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
    parser.add_argument('--gpumem', default=0.5, action='store', type=float, dest='gpumem', help='indicate the per process gpu memory fraction')
    parser.add_argument('--distrb', default=False, action='store_true', dest='distrb', help='whether to distribute data over multiple devices')
    parser.add_argument('--distbknd', default='nccl', action='store', dest='distbknd', help='distribute framework backend')
    parser.add_argument('--disturl', default='env://', action='store', dest='disturl', help='distribute framework url')
    parser.add_argument('--traindev', default=False, action='store_true', help='whether to use dev dataset for training')
    parser.add_argument('--noeval', default=False, action='store_true', help='whether to train only')
    parser.add_argument('--noschdlr', default=False, action='store_true', help='force to not use scheduler whatever the default setting is')
    parser.add_argument('--maxlen', default=128, action='store', type=int, dest='maxlen', help='indicate the maximum sequence length for each samples')
    parser.add_argument('--maxtrial', default=50, action='store', type=int, dest='maxtrial', help='maximum time to try')
    parser.add_argument('--droplast', default=False, action='store_true', help='whether to drop the last incompleted batch')
    parser.add_argument('--weight_class', default=False, action='store_true', help='whether to drop the last incompleted batch')
    parser.add_argument('--clswfac', default='1', type=str, help='whether to drop the last incompleted batch')
    parser.add_argument('--bert_outlayer', default='-1', type=str, dest='output_layer', help='indicate which layer to be the output of BERT model')
    parser.add_argument('--resume', action='store', dest='resume', help='resume training model file')
    parser.add_argument('--refresh', default=False, action='store_true', dest='refresh', help='refresh the trained model with newest code')
    parser.add_argument('--onto', help='ontology data')
    parser.add_argument('--pred', help='prediction file')
    parser.add_argument('--sent', default=False, action='store_true', dest='sent', help='whether to use location of labels to split the text into sentences')
    parser.add_argument('--datapath', help='location of dataset')
    parser.add_argument('-i', '--input', help='input dataset')
    parser.add_argument('--prvres', help='previous results for re-ranking')
    parser.add_argument('-u', '--task', default='hpo_entilement', type=str, dest='task', help='the task name [default: %default]')
    parser.add_argument('--model', default='bert', type=str, dest='model', help='the model to be validated')
    parser.add_argument('--encoder', dest='encoder', help='the encoder to be used after the language model: pool, s2v or s2s')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model file')
    parser.add_argument('--seed', help='manually set the random seed')
    parser.add_argument('-c', '--cfg', help='config string used to update the settings, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
    parser.add_argument('-m', '--method', default='rerank', help='main method to run')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', dest='verbose', help='display detailed information')
    args = parser.parse_args()

    # Logging setting
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Update config
    global_vars = globals()
    cfg_kwargs = {} if args.cfg is None else ast.literal_eval(args.cfg)
    args.cfg = cfg_kwargs
    _update_cfgs(global_vars, cfg_kwargs)

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

    # Random seed setting
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Format some arguments
    args.output_layer = list(map(int, args.output_layer.split(',')))
    args.output_layer = args.output_layer[0] if len(args.output_layer) == 1 else args.output_layer
    args.clswfac = list(map(float, args.clswfac.split(','))) if ',' in args.clswfac else float(args.clswfac)

    main()
