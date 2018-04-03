# Calculating some measures of linguistic complexity

This repository holds the code used for the workshop paper

- Çağrı Çöltekin and Taraka Rama (2018) [_Exploiting Universal
    Dependencies Treebanks for Measuring Morphosyntactic
    Complexity_](mlc2018.pdf)
    In: Proceedings of the Measuring Language Complexity workshop.
    ([bib](mlc2018.bib))

in the workshop
[Measuring Language Complexity (MLC)](http://www.christianbentz.de/MLC_index.html).


The code is written in python, with very few dependencies.
The file `mlc-morph.py` (type `./mlc-morph.py -h` for detailed usage
information) calculates the morphological measures,
and `postag_entropy.py` calculates the POS tag entropy
reported in the paper. 
Both scripts require a set of [UD](http://universaldependencies.org/) treebanks
in [CoNLL-U](http://universaldependencies.org/format.html) format
as input,
and output a tab-separated listing each measure for each input treebank.

Please contact the authors if you have any questions or comments.
