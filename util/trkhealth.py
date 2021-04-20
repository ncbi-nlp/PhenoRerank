import os, sys, json, copy, requests

import ftfy

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'trkhealth')
API_KEY = ''
SC=';;'


BASE_URL = 'https://knowledge.pryzm.health/api/cr/v1'


def annotext(text, ontos=[], max_trail=-1, interval=1):
	url = '%s/%s' % (BASE_URL, 'annotate')
	params = dict(text=text, apiKey=API_KEY)
	trail = 0
	while max_trail <= 0 or trail < max_trail:
	    try:
	        res = requests.post(url, data=params)
	        if res.status_code != 200:
	            print('Server errors!')
	        else:
	            res = res.json()
	            return [dict(id=r['term']['curie'].replace(':', '_'), loc=(r['startOffset'], r['endOffset']), text=r['term']['label'], negated=r['negated']) for r in res['data']]
	        time.sleep(interval)
	    except Exception as e:
	        print(e)
	        print(res)
	        import time
	        if interval > 0: time.sleep(interval)
	        trail += 1
	return res


if __name__ == '__main__':
	text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
	print([a['id'] for a in annotext(text, ontos=['HP'])])
