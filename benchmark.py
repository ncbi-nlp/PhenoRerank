import os, sys, copy, logging
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util import common, processor, obo, ncbo, monainit, doc2hpo, metamap, trkhealth

import urllib3
urllib3.disable_warnings()

MODULES_MAP = {'obo':obo, 'ncbo':ncbo, 'monainit':monainit, 'doc2hpo':doc2hpo, 'metamap':metamap, 'trkhealth':trkhealth}
ANNOT_KWARGS = {'monainit': {'includeAcronym':'true', 'includeCat':'phenotype'}}

args = {}


def annot(ids, texts, method, cache_path='.cache', pid=0):
    print('Processing documents: %s' % ','.join(map(str, ids)))
    sys.stdout.flush()
    if not os.path.exists(cache_path): os.mkdir(cache_path)
    preds = []
    unf_cached_fpath = os.path.join(cache_path, '%s%s' % (method, '_loc' if args.withloc else ''))
    unf_cached_df = pd.read_csv(unf_cached_fpath, index_col='id') if os.path.exists(unf_cached_fpath) else pd.DataFrame({})
    unf_cached_ids = set(unf_cached_df.index.tolist()) if len(unf_cached_df) > 0 else set([])
    try:
        cached_df = pd.read_csv('%s/%s%s_%i' % (cache_path, method, '_loc' if args.withloc else '', pid), index_col='id') if os.path.exists('%s/%s%s_%i' % (cache_path, method, '_loc' if args.withloc else '', pid)) else pd.DataFrame(dict(id=[], preds=[])).set_index('id')
    except Exception as e:
        print(e)
        cached_ids = set([])
    else:
        cached_ids = set(cached_df.index.tolist())
    for i, (doc_id, text) in enumerate(zip(ids, texts)):
        if doc_id in cached_ids:
            preds.append(cached_df.loc[doc_id]['preds'].split(';') if type(cached_df.loc[doc_id]['preds']) is str else [])
            continue
        elif doc_id in unf_cached_ids:
            preds.append(unf_cached_df.loc[doc_id]['preds'].split(';') if type(unf_cached_df.loc[doc_id]['preds']) is str else [])
        else:
            annot_res = [['%s|%s%i:%i' % (annot['id'], '%i-'%tid if len(text)>1 else '', np.amin(annot['loc']), np.amax(annot['loc'])) if args.withloc and 'loc' in annot else annot['id'] for annot in MODULES_MAP[method].annotext(sub_txt, ontos=['HP'], **ANNOT_KWARGS.setdefault(method, {}))] for tid, sub_txt in enumerate(text)]
            preds.append(list(set(common.flatten_list(annot_res))))
        if (i % 50 == 0 or i == len(ids)-1):
            cached_df = pd.DataFrame(dict(id=ids[:i+1], preds=[';'.join(map(str, x)) if len(x)>0 else '' for x in preds])).set_index('id')
            cached_df.to_csv('%s/%s_%i' % (cache_path, method, pid))
    return preds

def main():
    data_df = pd.read_csv(os.path.join(args.input, '%s.csv' % args.dataset), sep='\t', encoding='utf-8')
    true_lbs = [list(set(lbs.split(';'))) if type(lbs) is str else [] for lbs in data_df['labels']]
    ids, texts = zip(*[(doc_id, txt if type(txt) is list else ([txt] if txt is not np.nan else [])) for doc_id, txt in zip(data_df['id'], data_df['text'])])

    preds = annot(ids, texts, args.annotator, cache_path='.cache.%s' % args.dataset)
    pred_df = copy.deepcopy(data_df)
    pred_df['preds'] = [';'.join(pred) for pred in preds]
    pred_df.to_csv('%s_%s%s_preds.csv' % (args.dataset, args.annotator, '_loc' if args.withloc else ''), sep='\t', index=None, encoding='utf-8')

if __name__ == '__main__':
    # Logging setting
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Evaluate different annotator on various datasets.')
    parser.add_argument('annotator', choices=['obo', 'ncbo', 'monainit', 'doc2hpo', 'metamap', 'clinphen', 'ncr', 'trkhealth'], help='name of the annotator')
    parser.add_argument('dataset', choices=['biolarkgsc', 'copd'], help='path of the dataset file')
    parser.add_argument('-x', '--nlplib', choices=['nltk', 'spacy'], help='library for natural language processing')
    parser.add_argument('-i', '--input', default='./data', help='folder path that contains the dataset file')
    parser.add_argument('-l', '--withloc', default=False, action='store_true', help='whether annotate the mention location')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='display detailed information')
    parser.add_argument('--ncboapikey', help='API key for NCBO annotator')
    args = parser.parse_args()

    # Delayed import to speedup script loading
    if args.annotator == 'ncbo':
        ncbo.API_KEY = args.ncboapikey if args.ncboapikey else os.environ.setdefault('NCBO_APIKEY', '')
    elif args.annotator == 'clinphen':
        from util import clinphen
        MODULES_MAP['clinphen'] = clinphen
    elif args.annotator == 'ncr':
        from util import ncr
        MODULES_MAP['ncr'] = ncr

    # Configure NLP library
    args.nlplib = args.nlplib if args.nlplib else os.environ.setdefault('NLPLIB', 'nltk')
    processor.init_nlplib(args.nlplib)

    main()
