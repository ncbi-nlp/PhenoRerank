import os, io, re, sys, json, yaml, copy, codecs, itertools, signal, logging

from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix

import torch
from torch import nn

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from .dataset import DataParallel


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if np.isnan(metrics):
            return True
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True


def mkdir(path):
	if path and not os.path.exists(path):
		print(("Creating folder: " + path))
		os.makedirs(path)


def read_file(fpath, code='ascii'):
	if (isinstance(fpath, io.StringIO)):
		return fpath.readlines()
	try:
		data_str = []
		if (code.lower() == 'ascii'):
			with open(fpath, 'r') as fd:
				for line in fd.readlines():
					data_str.append(line)
		else:
			with codecs.open(fpath, mode='r', encoding=code, errors='ignore') as fd:
				for line in fd.readlines():
					data_str.append(line)
	except Exception as e:
		print(e)
		print(('Can not open the file \'%s\'!' % fpath))
		raise
	return data_str


def write_file(content, fpath, code='ascii'):
	if (isinstance(fpath, io.StringIO)):
		fpath.write(content)
		return
	try:
		if (code.lower() == 'ascii'):
			with open(fpath, mode='w') as fd:
				fd.write(content)
				fd.close()
		else:
			with codecs.open(fpath, mode='w', encoding=code, errors='ignore') as fd:
				fd.write(content)
				fd.close()
	except Exception as e:
		print(e)
		print(('Can not write to the file \'%s\'!' % fpath))
		sys.exit(-1)


def read_json(fpath):
	with open(fpath) as fd:
		res = json.load(fd)
	return res


def write_df(df, fpath, with_col=True, with_idx=False, sparse_fmt=None, compress=False):
	fs.mkdir(os.path.dirname(fpath))
	fpath = os.path.splitext(fpath)[0] + '.npz'
	if (compress):
		save_f = np.savez_compressed
	else:
		save_f = np.savez
	if (sparse_fmt == None or (type(sparse_fmt) == str and sparse_fmt.lower() == 'none')):
		save_f(fpath, data=df.values, shape=df.shape, col=df.columns.values if with_col else None, idx=df.index.values if with_idx else None)
	elif (sparse_fmt == 'csc'):
		sp_mt = sparse.csc_matrix(df.values)
		save_f(fpath, data=sp_mt.data, indices=sp_mt.indices, indptr=sp_mt.indptr, shape=sp_mt.shape, col=df.columns.values if with_col else None, idx=df.index.values if with_idx else None)
	elif (sparse_fmt == 'csr'):
		sp_mt = sparse.csr_matrix(df.values)
		save_f(fpath, data=sp_mt.data, indices=sp_mt.indices, indptr=sp_mt.indptr, shape=sp_mt.shape, col=df.columns.values if with_col else None, idx=df.index.values if with_idx else None)


def write_yaml(data, fpath, append=False, dfs=False):
	fpath = os.path.splitext(fpath)[0] + '.yaml'
	with open(fpath, 'a' if append else 'w') as f:
		yaml.dump(data, f, default_flow_style=dfs)


def read_yaml(fpath):
	fpath = os.path.splitext(fpath)[0] + '.yaml'
	if (os.path.exists(fpath)):
		with open(fpath, 'r') as f:
			return yaml.load(f)
	else:
		print(('File %s does not exist!' % fpath))


def param_reader(fpath):
	data = read_yaml(fpath)
	def get_params(mdl_t, mdl_name, data=data):
		if (not data):
			print(('Cannot find the config file: %s' % fpath))
			return {}
		if (mdl_t in data):
			mdl_list = data[mdl_t]
		else:
			print(('Model type %s does not exist.' % mdl_t))
			return {}
		for mdl in mdl_list:
			if (mdl['name'] == mdl_name):
				return mdl['params']
		else:
			print(('Parameters of model %s does not exist.' % mdl_name))
			return {}
	return get_params


def cfg_reader(fpath):
	data = read_yaml(fpath)
	def get_params(module, function, data=data):
		if (not data):
			print(('Cannot find the config file: %s' % fpath))
			return {}
		if (module in data):
			func_list = data[module]
		else:
			print(('Module %s does not exist.' % module))
			return {}
		for func in func_list:
			if (func['function'] == function):
				return func['params']
		else:
			print(('Parameters of function %s does not exist.' % func))
			return {}
	return get_params


def gen_pytorch_wrapper(mdl_type, mdl_name, **kwargs):
    from . import config as C
    wrapper_cls = PytorchSeq2SeqWrapper if mdl_type == 'seq2seq' else PytorchSeq2VecWrapper
    mdl_cls = C.PYTORCH_WRAPPER[mdl_name]
    return wrapper_cls(module=mdl_cls(**kwargs))


def gen_mdl(config, use_gpu=False, distrb=False, dev_id=None):
    mdl_name, pretrained = config.model, True if type(config.pretrained) is str and config.pretrained.lower() == 'true' else config.pretrained
    if mdl_name == 'none': return None, None
    wsdir = config.wsdir if hasattr(config, 'wsdir') and os.path.isdir(config.wsdir) else '.'
    common_cfg = config.common_cfg if hasattr(config, 'common_cfg') else {}
    pr = param_reader(os.path.join(wsdir, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    params = pr('LM', config.lm_params)
    lm_config = config.lm_config(**params)
    if distrb: import horovod.torch as hvd
    if (type(pretrained) is str):
        if (not distrb or distrb and hvd.rank() == 0): logging.info('Using pretrained model from `%s`' % pretrained)
        checkpoint = torch.load(pretrained, map_location='cpu')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
    elif (pretrained):
        if (not distrb or distrb and hvd.rank() == 0): logging.info('Using pretrained model...')
        mdl_name = mdl_name.split('_')[0]
        model = config.lm_model.from_pretrained(params['pretrained_mdl_path'] if 'pretrained_mdl_path' in params else config.lm_mdl_name)
    else:
        if (not distrb or distrb and hvd.rank() == 0): logging.info('Using untrained model...')
        try:
            for pname in ['pretrained_mdl_path', 'pretrained_vocab_path']:
                if pname in params: del params[pname]
            if (mdl_name == 'elmo'):
                pos_params = [lm_config[k] for k in ['options_file','weight_file', 'num_output_representations']]
                kw_params = dict([(k, lm_config[k]) for k in lm_config.keys() if k not in ['options_file','weight_file', 'num_output_representations', 'elmoedim']])
                logging.info('ELMo model parameters: %s %s' % (pos_params, kw_params))
                model = config.lm_model(*pos_params, **kw_params)
            else:
                model = config.lm_model(lm_config)
        except Exception as e:
            logging.warning(e)
            logging.warning('Cannot find the pretrained model file, using online model instead.')
            model = config.lm_model.from_pretrained(config.lm_mdl_name)
    if (use_gpu): model = model.to('cuda')
    return model, lm_config


def gen_clf(config, lm_model, lm_config, use_gpu=False, distrb=False, dev_id=None, **kwargs):
    mdl_name, constraints = config.model, config.cnstrnts.split(',') if hasattr(config, 'cnstrnts') and config.cnstrnts else []
    lm_mdl_name = mdl_name.split('_')[0]
    kwargs.update(dict(config=config, lm_model=lm_model, lm_config=lm_config))
    common_cfg = config.common_cfg if hasattr(config, 'common_cfg') else {}
    wsdir = config.wsdir if hasattr(config, 'wsdir') and os.path.isdir(config.wsdir) else '.'
    pr = param_reader(os.path.join(wsdir, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    params = pr('LM', config.lm_params) if lm_mdl_name != 'none' else {}
    for pname in ['pretrained_mdl_path', 'pretrained_vocab_path']:
        if pname in params: del params[pname]

    lvar = locals()
    for x in constraints:
        cnstrnt_cls, cnstrnt_params = copy.deepcopy(C.CNSTRNTS_MAP[x])
        constraint_params = pr('Constraint', C.CNSTRNT_PARAMS_MAP[x])
        cnstrnt_params.update(dict([((k, p), constraint_params[p]) for k, p in cnstrnt_params.keys() if p in constraint_params]))
        cnstrnt_params.update(dict([((k, p), kwargs[p]) for k, p in cnstrnt_params.keys() if p in kwargs]))
        cnstrnt_params.update(dict([((k, p), lvar[p]) for k, p in cnstrnt_params.keys() if p in lvar]))
        kwargs.setdefault('constraints', []).append((cnstrnt_cls, dict([(k, v) for (k, p), v in cnstrnt_params.items()])))

    clf = config.clf[config.encoder](**kwargs) if hasattr(config, 'embed_type') and config.embed_type else config.clf(**kwargs)
    if use_gpu: clf = _handle_model(clf, dev_id=dev_id, distrb=distrb)
    return clf


def save_model(model, optimizer, fpath='checkpoint.pth', in_wrapper=False, devq=None, distrb=False, **kwargs):
    logging.info('Saving trained model...')
    use_gpu, multi_gpu = (devq and len(devq) > 0), (devq and len(devq) > 1)
    if not distrb and (in_wrapper or multi_gpu): model = model.module
    model = model.cpu() if use_gpu else model
    checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer':optimizer if not distrb else None, 'optimizer_state_dict':optimizer.state_dict()}
    checkpoint.update(kwargs)
    torch.save(checkpoint, fpath)
    model = _handle_model(model, dev_id=devq, distrb=distrb) if use_gpu else model


def load_model(mdl_path):
    logging.info('Loading previously trained model...')
    checkpoint = torch.load(mdl_path, map_location='cpu')
    model, optimizer = checkpoint['model'], checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint.setdefault('resume', {}), dict([(k, v) for k, v in checkpoint.items() if k not in ['model', 'state_dict', 'optimizer', 'optimizer_state_dict', 'resume']])


def _weights_init(mean=0., std=0.02):
    def _wi(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(mean, std)
        elif classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.normal_(m.bias, 0)
    return _wi


def _sprmn_cor(trues, preds):
    return sp.stats.spearmanr(trues, preds)[0]


def _prsn_cor(trues, preds):
    return sp.stats.pearsonr(trues, preds)[0]


def _handle_model(model, dev_id=None, distrb=False):
    if (distrb):
        model.cuda()
    elif (dev_id is not None):
        if (type(dev_id) is list):
            model = model.to('cuda')
            model = DataParallel(model, device_ids=dev_id)
        else:
            torch.cuda.set_device(dev_id)
            model = model.to('cuda')
    return model


def _update_cfgs(global_vars, cfgs):
    for glb, glbvs in cfgs.items():
        if glb in global_vars:
            if type(global_vars[glb]) is dict:
                for cfgk in [opts.task, opts.model, opts.method]:
                    if cfgk in global_vars[glb]:
                        if type(global_vars[glb][cfgk]) is dict:
                            global_vars[glb][cfgk].update(glbvs)
                        else:
                            global_vars[glb][cfgk] = glbvs
            else:
                global_vars[glb] = glbvs


def normalize(a, ord=1):
    norm=np.linalg.norm(a, ord=ord)
    if norm==0: norm=np.finfo(a.dtype).eps
    return a/norm


def flatten_list(nested_list):
	if not hasattr(nested_list, '__iter__') or isinstance(nested_list, str): return nested_list
	l = list(itertools.chain.from_iterable(x if hasattr(x, '__iter__') and not isinstance(x, str) else [x] for x in nested_list))
	if (len(l) == 0): return []
	if (any([type(j) is list for j in l])):
		return flatten_list(l)
	else:
		return l
