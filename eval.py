import os, sys, ast, random, logging
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, zero_one_loss, confusion_matrix

from util.config import *
from util.common import _update_cfgs, flatten_list

global FILE_DIR, DATA_PATH, args
FILE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
args = {}


def exmp_accuracy(y_true, y_pred):
    y_true, y_pred = y_true.astype('bool').astype('int8'), y_pred.astype('bool').astype('int8')
    return ((y_true * y_pred).sum(axis=1) / (np.finfo(float).eps + (y_true + y_pred).astype('bool').astype('int8').sum(axis=1))).mean()

def exmp_precision(y_true, y_pred):
    y_true, y_pred = y_true.astype('bool').astype('int8'), y_pred.astype('bool').astype('int8')
    return ((y_true * y_pred).sum(axis=1) / (np.finfo(float).eps + y_pred.sum(axis=1))).mean()

def exmp_recall(y_true, y_pred):
    y_true, y_pred = y_true.astype('bool').astype('int8'), y_pred.astype('bool').astype('int8')
    return ((y_true * y_pred).sum(axis=1) / (np.finfo(float).eps + y_true.sum(axis=1))).mean()

def exmp_fscore(y_true, y_pred, beta=1):
    precision = exmp_precision(y_true, y_pred)
    recall = exmp_recall(y_true, y_pred)
    beta_pw2 = beta**2
    return (1 + beta_pw2)*(precision * recall)/(beta_pw2 * precision + recall)


def default():
    global args
    data_df = pd.read_csv(os.path.join(DATA_PATH, args.dataset+'.csv'), sep='\t', dtype={'id': str}, encoding='utf-8')
    true_lbs = [list(set(lbs.split(';'))) if type(lbs) is str else [] for lbs in data_df['labels']]
    methods = ['_'.join([s for s in os.path.splitext(os.path.basename(pred))[0].split('_') if s not in [args.dataset, 'preds']]) for pred in args.preds]
    pred_dfs = [pd.read_csv(pred, sep='\t', dtype={'id': str}, encoding='utf-8') for pred in args.preds]
    preds_list = [[list(set(y.split(';'))) if type(y) is str else[] for y in df['preds']] for df in pred_dfs]
    if args.withloc: preds_list = [[list(set([lb.split('|')[0] for lb in y])) for y in preds] for preds in preds_list]

    outputs = []
    for i, (method, preds) in enumerate(zip(methods, preds_list)):
        orig_true_lbs = true_lbs
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(true_lbs + preds)
        lbidx = [np.where(mlb.classes_ == x)[0].item() for x in pd.Series(flatten_list(true_lbs)).value_counts().sort_values(ascending=False).index.tolist()]
        lbs = mlb.transform(true_lbs)
        pred_lbs = mlb.transform(preds)
        perf_df = pd.DataFrame(classification_report(lbs, pred_lbs, labels=lbidx, target_names=[mlb.classes_[x] for x in lbidx], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
        perf_df.to_excel('perf_%s_%s.xlsx' % (args.dataset, method))
        most_lbidx = lbidx[:args.topn]
        topn_perf_df = pd.DataFrame(classification_report(lbs, pred_lbs, labels=most_lbidx, target_names=[mlb.classes_[x] for x in most_lbidx], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
        topn_perf_df.to_excel('top%i_perf_%s_%s.xlsx' % (args.topn, args.dataset, method))
        tns, fps, fns, tps = confusion_matrix(lbs.ravel(), pred_lbs.ravel()).ravel()
        print(method + '\t accuracy: %.4f' % exmp_accuracy(lbs, pred_lbs) + '\t hamming loss: %.4f' % hamming_loss(lbs, pred_lbs) + '\t 0/1 loss: %.4f' % zero_one_loss(lbs, pred_lbs))
        print('exmp-precision: %.4f' % exmp_precision(lbs, pred_lbs) + '\texmp-recall: %.4f' % exmp_recall(lbs, pred_lbs) + '\texmp-f1-score: %.4f' % exmp_fscore(lbs, pred_lbs))
        outputs.append((exmp_precision(lbs, pred_lbs), exmp_recall(lbs, pred_lbs), exmp_fscore(lbs, pred_lbs), tps, fns, fps, tns))
        print('tns, fps, fns, tps: %s' % ', '.join(map(str, [tns, fps, fns, tps])) + '\n')
        print(topn_perf_df)
        print('\n\n')
        true_lbs = orig_true_lbs
        
    precision, recall, fscore, true_pos, false_neg, false_pos, true_neg = zip(*outputs)
    pd.DataFrame(dict(method=methods, precision=precision, recall=recall, fscore=fscore, tp=true_pos, fn=false_neg, fp=false_pos, tn=true_neg)).to_excel('outputs.xlsx')


def main():
    if args.method == 'default':
        main_func = default
    main_func()

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Evaluate the results.')
    parser.add_argument('dataset', choices=['biolarkgsc', 'copd'], help='dataset name')
    parser.add_argument('preds', nargs='+', help='prediction files')
    parser.add_argument('-i', '--input', default='./data', help='folder path that contains the dataset file')
    parser.add_argument('-l', '--withloc', default=False, action='store_true', help='whether annotate the mention location')
    parser.add_argument('-k', '--topn', default=10, type=int, help='whether annotate the mention location')
    parser.add_argument('-c', '--cfg', help='config string used to update the settings, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
    parser.add_argument('-m', '--method', default='default', help='main method to run')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', dest='verbose', help='display detailed information')
    args = parser.parse_args()

    # Logging setting
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Update config
    global_vars = globals()
    cfg_kwargs = {} if args.cfg is None else ast.literal_eval(args.cfg)
    args.cfg = cfg_kwargs
    _update_cfgs(global_vars, cfg_kwargs)

    # Process config
    if args.input is not None: DATA_PATH = args.input

    main()
