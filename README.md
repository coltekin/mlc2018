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

## MLC 2019 changes additions

We add only a single measure, _morphological inflection accuracy_,
for the [MLC 2019 workshop](http://www.christianbentz.de/MLC2019_index.html).
The measure is simply based on a simple morphological inflection system,
that tries to inflect a given lemma based on the features in the treebank.
We use the average accuracy of inflection system in a k-fold (3-fold by default)
cross validation setting as a measure of morphological complexity
(the script also computes mean edit distance, MED).
The present model uses an SVM classifier to predict transducer actions
(the model is described in [this paper](https://www.aclweb.org/anthology/W19-4209)).
However, any other inflection model can be used for calculating the measure.

The name of the script for training and testing is `morph-inflect.py`.
The script requires basic Python machine learning libraries (sklearn, numpy).
If run with the default `-c tune` option,
it runs a random search over indicated parameter range and outputs
the accuracy and MED for each setting.
Tuning on all treebanks may require considerable computation time.
If run with the `-c test` option,
it outputs the cross-validation scores with the given parameters,
or optionally reading them from a file.
An example parameter file (after a few days of random search)
for all MLC 2019 treebanks with a sample size of 1000
and 3-fold CV is provided as `best_params.s1000`.

Please contact the authors if you have any questions or comments.
