# PhenoRerank
`PhenoRerank` contains the source code and pre-processed datasets for benchmarking the phenotype annotators and re-ranking the results. To facilitate the benchmarking, we provide the wrapper code of several existing annotators including OBO, NCBO, Monarch Initiative, Doc2hpo, MetaMap, Clinphen, NeuralCR, TrackHealth. We developed a re-ranking model that can boost the performance of the annotators in particular for precision. It filters out the false positives based on the contextual information. It is pre-trained on the pretext task defined on the textual data in Human Phenotype Ontology (i.e. term names, synonyms, definitions, and comments). It can also be fine-tuned on a specific dataset for further improvement.


## Getting Started
The following instructions will help you setup the programs as well as the datasets, and re-produce the benchmarking results.

### Prerequisities
Firstly, you need to install a Python Interpreter (tested 3.6.10) and the following packages:
* numpy (tested 1.18.5)
* pandas (tested 1.0.5)
* ftfy (tested 5.7)
* apiclient (tested 1.0.4)
* pymetamap (tested 0.2)
* clinphen (tested 1.28)
* rdflib \[optional\] \(tested 4.2.2\)

### Download the external programs
* Run the script `install.sh` to download and configure the external programs for benchmark.
* Follow the instructions [here](https://metamap.nlm.nih.gov/Installation.shtml) to install MetaMap and make sure that the locations of programs `skrmedpostctl` and `wsdserverctl` are added to `$PATH`
* Follow the instructions [here](https://github.com/ccmbioinfo/NeuralCR#installation) to install the dependencies of NeuralCR and download the model parameters. Then make a copy or create a soft link of the `model_params` in the folder you are going to run benchmark.

### Obtain the API keys for some online tools
Follow the guidelines to get the API keys for [NCBO](http://www.bioontology.org/wiki/index.php/BioPortal_Help#Getting_an_API_key) and [TrackHealth](https://track.health/api/). Then assign to the `API_KEY` global variable in the wrapper `util/ncbo.py` and `util/trkhealth.py`.

### Locate the Pre-Generated Dataset
After cloning the repository and configuring the programs, you can download the pre-generated datasets and pre-trained model [here](https://www.doi.org/10.17632/v4t59p8w4z).

Filename | Description  
--- | ---
biolarkgsc.csv | Pre-processed BiolarkGSC+ dataset with document-level annotations
biolarkgsc_locs.csv | Pre-processed BiolarkGSC+ dataset with mention-level annotations
copd.csv | Pre-processed COPD-HPO dataset with document-level annotations
copd_locs.csv | Pre-processed COPD-HPO dataset with mention-level annotations

You can load a dataset into a [Pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) using the following code snippet.
```python
import pandas as pd
data = pd.read_csv('XXX.csv', sep='\t', dtype={'id': str}, encoding='utf-8')
```

### A Simple Example
You can benchmark annotator `ncbo` on `biolarkgsc` dataset using the following command:
```bash
python benchmark.py ncbo biolarkgsc -i ./data
```
This command will search the dataset file `biolarkgsc.csv` in the path `./data` and output the result of annotator `ncbo` in the file `biolarkgsc_ncbo_preds.csv`

## Re-rank the result
Please download the pre-trained model `hpo_bert_onto.pth` or copy yours to the working folder in advance. Also, prepare the pre-processed HPO dictionary file `hpo_labels.csv` and your prediction file in the same folder where you run the following command.
```bash
python rerank.py --model bert_onto -u biolarkgsc --onto hpo_labels.csv --resume hpo_bert_onto.pth
```

## Evaluation
Once the prediction files are ready, please rename them appropriately. Then you can evaluate the results for comparison using the following commands.
```bash
python eval.py method1.csv method2.csv method3.csv
```

## Fine-tuning
For the sake of the best performance, you can fine-tune the re-ranking model on your own dataset if the dataset has sentence-/mention-level annotations. Use the following commands to firstly convert the dataset into appropriate format for training the re-ranking model.
```bash
python rerank.py -m train --noeval --model bert_onto --pretrained true -u biolarkgsc -f csv --onto hpo_labels.csv --pooler none --pdrop 0.1 --do_norm --norm_type batch --initln --earlystop --lr 0.0002 --maxlen 384 -j 10 -z 8 -g 0
```


## Dataset Re-Generation

You can re-generate the dataset from the annotations of [BiolarkGSC+](https://github.com/lasigeBioTM/IHP) and [COPD](http://www.nactem.ac.uk/COPD) using the following command:

```bash
python gendata.py -u biolarkgsc
python gendata.py -u copd
```
