import os, sys, json, copy, site
from io import StringIO

import pandas as pd

import ftfy

from clinphen_src import get_phenotypes, src_dir

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'clinphen')
SC=';;'


def annotext(text, ontos=[], umls=False, rare_pheno=False):
	custom_thesaurus = os.path.join(clinphenext.srcDir, 'data', 'hpo_umls_thesaurus.txt') if umls else ''
	df = pd.read_table(StringIO(clinphenext(StringIO(ftfy.fix_text(text)), custom_thesaurus, rare_pheno)))
	res = [dict(zip(df.columns, x)) for x in zip(*[df[col] for col in df.columns])]
	return [dict(id=r['HPO ID'].replace(':', '_'), name=r['Phenotype name'], occrns=r['No. occurrences'], text=r['Example sentence']) for r in res]

## Reference code start ##
srcDir = os.path.join([s for s in site.getsitepackages() if 'site-packages' in s][0], 'clinphen_src')
myDir = "/".join(os.path.realpath(__file__).split("/")[:-1])

def load_common_phenotypes(commonFile):
  returnSet = set()
  for line in open(commonFile): returnSet.add(line.strip())
  return returnSet

def clinphenext(inputFile, custom_thesaurus="", rare=False):
  hpo_main_names = os.path.join(srcDir, "data", "hpo_term_names.txt")

  def getNames():
    returnMap = {}
    for line in open(hpo_main_names):
      lineData = line.strip().split("\t")
      returnMap[lineData[0]] = lineData[1]
    return returnMap
  hpo_to_name = getNames()

  inputStr = ""
  for line in inputFile.readlines() if type(inputFile) is StringIO else open(inputFile): inputStr += line
  if not custom_thesaurus: returnString = get_phenotypes.extract_phenotypes(inputStr, hpo_to_name)
  else: returnString = get_phenotypes.extract_phenotypes_custom_thesaurus(inputStr, custom_thesaurus, hpo_to_name)
  if not rare: return returnString
  items = returnString.split("\n")
  returnList = []
  common = load_common_phenotypes(os.path.join(srcDir, "data", "common_phenotypes.txt"))
  for item in items:
    HPO = item.split("\t")[0]
    if HPO in common: continue
    returnList.append(item)
  return "\n".join(returnList)
## Reference code end ##


if __name__ == '__main__':
    text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    print([a['id'] for a in annotext(text, ontos=['HP'])])
