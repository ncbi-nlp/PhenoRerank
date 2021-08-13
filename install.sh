#!/bin/bash

mkdir -p ext && cd ext

# OBO
if [ ! -d "obo" ]; then
	rm -rf dist
	if [ ! -f "OBOAnnotatorNoGUI.zip" ]; then
		wget https://www.usc.gal/keam/PhenotypeAnnotation/OBOAnnotatorNoGUI.zip
	fi
	unzip OBOAnnotatorNoGUI.zip
	mv dist obo
	rm -f OBOAnnotatorNoGUI.zip
fi

# ClinPhen
if [ ! -d "clinphen" ]; then
	git clone https://dawnyesky@bitbucket.org/bejerano/clinphen.git
fi

# NeuralCR
if [ ! -d "NeuralCR" ]; then
	git clone https://github.com/ccmbioinfo/NeuralCR.git
fi

cd ..

echo -e "# Added by PhenoRerank on $(date)
export OBO_HOME=$(pwd)/ext/OBOAnnotatorNoGUI/dist
export CLASSPATH=\$OBO_HOME/OBOAnnotatorNoGUI.jar:$CLASSPATH
export PYTHONPATH=$(pwd)/ext/clinphen:$PYTHONPATH
export PYTHONPATH=$(pwd)/ext/NeuralCR:$PYTHONPATH
" >> ~/.bashrc