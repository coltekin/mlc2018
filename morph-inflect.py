#!/usr/bin/env python3

import sys, os
import argparse
import numpy as np
from align import simple_align
from mlc_data import read_treebank
from sklearn.svm import LinearSVC, SVC 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
import itertools
import editdistance
import random
from hashlib import md5
from aligner import Aligner, print_alignment
from multiprocessing import Pool

def random_iter(param_space, max_reject=1000, max_iter=None):
    """ A generator that returns random drwas from a parameter space.

        param_space is a sequence (name, type, range)
        where type is either 'numeric' or 'categorical',
        and range is a triple (start, stop, step) for 
        numeric parameters, and another sequence of 
        parameter values to explore.

        the function keeps the hashes of returned parameter values, and
        if an already returned parameter set is drawn max_iter times,
        it terminates.
    """
    seen = set()
    rejected = 0
    while True:
        params = []
        for param, type_, seq in param_space:
            if 'numeric'.startswith(type_):
                try:
                    start, stop, step = seq
                    p_range = np.arange(start, stop + step, step).tolist()
                except:
                    start, stop = seq
                    p_range = np.arange(start, stop + 1, 1).tolist()
                rval = random.choice(p_range)
                params.append((param, rval))
            elif 'categorical'.startswith(type_):
                params.append((param, random.choice(seq)))
        param_hash = md5(str(params).encode()).digest()
        if param_hash not in seen:
            seen.add(param_hash)
            rejected = 0
            yield dict(params)
        else:
            rejected += 1
            if rejected == max_reject:
                print("More than {} iterations with already drawn parameters. "
                      "The search space is probably exhausted".format(max_reject),
                      file=sys.stderr)
                return
        if max_iter and max_iter <= len(seen):
            return

def count_gaps(seq):
    gaps = 0
    prev = None
    for s in seq:
        if s in {'', '<', '>'}  and prev != '':
            gaps += 1
        prev = s
    return gaps

class DummyClassifier:
    def __init__(self, **kwargs):
        self.output = ""
    def fit(self, x, y, **kwargs):
        if len(set(y)) == 1:
            self.output = y[0]
    def predict(self, x, **kwargs):
        return np.full((x.shape[0],), self.output)

class Inflection:
    _PARAMS = {'C': 1.0, 'window': 3,
            'cross_features': 2, 'classifier': 'mono',
            'C_replace': 0.0, 'C_insert': 0.0}
    def __init__(self, feature_type, **params):
        self.feature_type = feature_type
        self.a = Aligner(method='lcs')
        if feature_type == "sparse":
            self.get_features = self.get_sparse_features
        elif feature_type == "sparse2":
            self.get_features = self.get_sparse2_features
        elif feature_type == "onehot":
            self.get_features = self.get_positional_features
        self.old_window = None
        self.sample_weight = None
        self._reset()
        self.set_params(**params)

    def _reset(self):
        for k, v in self._PARAMS.items():
            setattr(self, k, v)

    def set_params(self, **params):
        for k, v in params.items():
            if k in self._PARAMS:
                setattr(self, k, v)

    def vectorize(self, lem, tag, wf=None): 
        if self.old_window != self.window:
            self.old_window = self.window
            self.features = []
            self.labels = []
            for i, (l, t, w) in enumerate(zip(lem, tag, wf)):
                alignments = self.a.align(l, w)
                alignments = [[('<','<')] + x + [('>','>')] for x in alignments]
                
                for j, a in enumerate(alignments):
                    if j > 0: break # in case there are multiple alignments take only the first
                    li, wi = 0, 0
                    for k,(lc, wc) in enumerate(a):
                        self.features.append(
                                self.get_features('<' + l + '>', '<' + w[:wi], t, li,
                                    window=(self.window, self.window)))
                        if lc == '':
                            action = 'insert:' + wc
                            wi += 1
                        elif lc == wc:
                            action = 'copy:'
                            li += 1
                            wi += 1
                        elif wc == '':
                            action = 'delete:'
                            li += 1
                        else:
                            action = 'replace:' + wc
                            li += 1
                            wi += 1
                        self.labels.append(action)

            if self.feature_type.startswith('sparse'):
                self.vec = TfidfVectorizer(sublinear_tf=True,
                        analyzer=lambda x: x)
                self.x = self.vec.fit_transform(self.features)
            else:
                self.x = np.array(self.features)

    def fit(self, wf, lem, tag):
        print("vecorize....", file=sys.stderr)
        self.vectorize(lem, tag, wf)
        print(self.x.shape, file=sys.stderr)

        print("fit....", file=sys.stderr)
        if self.classifier == 'twostep':
            action = [s.split(':')[0] for s in self.labels]
            self.clf = LinearSVC(C=self.C, class_weight='balanced',
                    max_iter=1000)
            self.clf.fit(self.x, action, sample_weight=self.sample_weight)

            replace_i = [i for i in range(len(self.labels))\
                    if self.labels[i].startswith('replace')]
            sw = None
            if len(replace_i):
                x = self.x[replace_i, :]
                y = np.array(self.labels)[replace_i]
                if self.C_replace == 0.0: self.C_replace = self.C
                if len(set(y)) == 1:
                    self.clf_replace = DummyClassifier()
                else:
                    self.clf_replace = LinearSVC(C=self.C_replace,
                            class_weight='balanced', max_iter=50000)
                self.clf_replace.fit(x, y)
            else:
                self.clf_replace = DummyClassifier()

            insert_i = [i for i in range(len(self.labels))\
                    if self.labels[i].startswith('insert')]
            if len(insert_i):
                x = self.x[insert_i, :]
                y = np.array(self.labels)[insert_i]
                if self.C_insert == 0.0: self.C_insert = self.C
                if len(set(y)) == 1:
                    self.clf_replace = DummyClassifier()
                else:
                    self.clf_insert = LinearSVC(C=self.C_replace,
                            class_weight='balanced', max_iter=50000)
                self.clf_insert.fit(x, y)
            else:
                self.clf_insert = DummyClassifier()
        else:
            self.clf = LinearSVC(C=self.C, class_weight='balanced',
                    max_iter=50000)
            self.clf.fit(self.x, self.labels, sample_weight=self.sample_weight)

    def predict(self, x):
        if self.classifier == 'twostep':
            action = str(self.clf.predict(x)[0])
            ch = ''
            if action == 'insert':
                if self.clf_insert is None:
                    action = 'copy'
                else:
                    ch = str(self.clf_insert.predict(x)[0]).split(':', 1)[1]
            elif action == 'replace':
                if self.clf_replace is None:
                    action = 'copy'
                else:
                    ch = str(self.clf_replace.predict(x)[0]).split(':', 1)[1]
            return action, ch
        else:
            return str(self.clf.predict(x)[0]).split(':', 1)

    def decode(self, lemma, tags, max_len=30):
        w_prefix = ''
        li = 0
        while li < len(lemma):
            feat = self.get_features(lemma, w_prefix, tags, li)
            if self.feature_type.startswith('sparse'):
                x = self.vec.transform([feat])
            else:
                x = np.array([feat])
            act, arg = self.predict(x)
            if act == 'copy':
                w_prefix +=  lemma[li]
                li += 1
            elif act == 'replace':
                w_prefix += arg
                li += 1
            elif act == 'insert':
                w_prefix += arg
            elif act == 'delete':
                li += 1
            if len(w_prefix) > max_len or w_prefix and w_prefix[-1] == '>':
                break
        return w_prefix

    def get_sparse_features(self, lemma, word_prefix, tags, idx, window=(10,10)):
        cross = self.cross_features
        pfx_feat, sfx_feat, wpfx_feat = [], [], []
        tag_feat = ["tag:{}".format(t) for t in tags]
        if cross >= 2:
            tag_feat += ["tag2:{}-{}".format(t,t) for t in
                    itertools.product(tags,tags)]
        ch_feat = ["ch:{}".format(lemma[idx])]
        for i in range(1,window[0]+1):
            if i <= idx:
                pfx_feat.append('lprefix:{}'.format(lemma[idx-i:idx]))
            if i <= len(word_prefix):
                wpfx_feat.append('wprefix:{}'.format(word_prefix[-i:]))
        for i in range(idx+1,idx+window[1]):
            if i <= len(lemma):
                sfx_feat.append('lsuffix:{}'.format(lemma[idx:i]))
        str_feat = ch_feat + pfx_feat + sfx_feat + wpfx_feat
        if cross > 3:
            cross = ["&".join((x,y)) for x,y in
                    itertools.product(pfx_feat,sfx_feat)]
            cross = ["&".join((x,y)) for x,y in
                    itertools.product(wpfx_feat,cross)]
            cross = ["&".join((x,y)) for x,y in
                    itertools.product(ch_feat,cross)]
            str_feat += cross
        else:
            cross = ["&".join((x,y)) for x,y in
                    itertools.product(ch_feat,str_feat)]
            str_feat += cross
        return str_feat + tag_feat + ["&".join((x, y)) for x, y in itertools.product(tag_feat,str_feat)]

    def get_sparse2_features(self, lemma, word_prefix, tags, idx, window=(10,10)):
        cross = self.cross_features
        pfx_feat, sfx_feat, wpfx_feat = [], [], []
        tag_feat = [{"t:{}".format(t)} for t in tags]
        ch_feat = [{"l0:{}".format(lemma[idx])}]
        for i in range(1,window[0] + 1):
            if i <= idx:
                pfx_feat.append({'l-{}:{}'.format(i, lemma[idx-i])})
            if i <= len(word_prefix):
                wpfx_feat.append({'w-{}:{}'.format(i, word_prefix[-i])})
        for i in range(1,window[1] + 1):
            if (idx + i) < len(lemma):
                sfx_feat.append({'l+{}:{}'.format(i, lemma[idx+i])})
        str_feat = ch_feat + pfx_feat + sfx_feat + wpfx_feat
        feat = str_feat + tag_feat
        feat_cross = feat
        for i in range(cross):
            feat_cross = [x|y for x, y in itertools.product(feat, feat_cross)]
        return ['&'.join(sorted(f)) for f in feat_cross]

    def get_positional_features(self, lemma, word_prefix, tags, idx, window=(3,3)):
        chars = [lemma[idx]]
        tag_enc = self.data.te
        ch_enc = self.data.ce
        for i in range(idx - (window[0] + 1), idx - 1):
            if i >= 0:
               chars.append(lemma[i])
               chars.append(word_prefix[i])
            else:
               chars.append(ch_enc.pad)
        for i in range(idx+1,idx+window[1]+1):
            if i < len(lemma):
               chars.append(lemma[i])
            else:
               chars.append(ch_enc.pad)
        feat = np.array(ch_enc.encode(chars, onehot=True)).flatten()
        feat = np.concatenate((feat, tag_enc.transform([tags])[0]))
        return feat

    def evaluate(self, wf, lemmas, tags):
        acc = 0
        med = 0
        for i, word in enumerate(wf):
            tag = tags[i]
            lem = lemmas[i]
            pred = self.decode(lem, tag) 
#            print(word, pred, file=sys.stderr) 
            acc += int(pred == word)
            med += editdistance.eval(pred, word)
        med = med / len(wf)
        acc = acc / len(wf)
        print(acc, med, file=sys.stderr)
        return(acc, med)


def train_test(params, k=3, max_size=None, feature_type='sparse',
        cross_features=2, classifier='mono'):
    print("Reading the treebank", file=sys.stderr)
    d = read_treebank(params['treebank'], shuffle=True, max_size=max_size)
    print(len(d),file=sys.stderr)
    d = np.array(d, dtype=object)
    print(d.shape, file=sys.stderr)

    print("Initializing the model", file=sys.stderr)
    m = Inflection(feature_type=feature_type,
            cross_features=cross_features,
            classifier=classifier,
            **params)

    kf = KFold(n_splits=k)
    kf.get_n_splits(d)

    scores = []
    for i, (ti, vi) in enumerate(kf.split(d)):
        wf_trn, lem_trn, tag_trn = d[ti].T
        wf_val, lem_val, tag_val = d[vi].T
        print("Fold {}: fit".format(i), file=sys.stderr)
        m.fit(wf_trn, lem_trn, tag_trn)
        print("Fold {}: evaluate".format(i), file=sys.stderr)
        scores.append(m.evaluate(wf_val, lem_val, tag_val))
    scores = np.array(scores)
    return scores

class FitFunction():
    """ Function class to get around limiteations of Pool.
    """
    def __init__(self, target_only=False, feature_type='sparse',
            cross_features=2, classifier='mono', data_size=None):
        self.target_only = target_only
        self.feature_type = feature_type
        self.cross_features = cross_features
        self.classifier = classifier
        self.k = 3
        self.data_size = data_size
    def __call__(self, p):
        print(p, file=sys.stderr)
        scores = train_test(p, k=self.k, max_size=self.data_size,
                    feature_type=self.feature_type,
                    cross_features=self.cross_features,
                    classifier=self.classifier)
        print('|', p, scores[:,0].mean(), scores[:,0].std(),
                 scores[:,1].mean(), scores[:,1].std(), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('treebanks', nargs="+",
            help="A list of directories containing the treebanks. "
                 "All .conllu files under the directory will be used for tuning.")
    ap.add_argument('--command', '-c', choices=("tune","test","score"), default="tune")
    ap.add_argument('--params', '-p',
            help="Default parameters for testing (Key=Value pairs), or "
                 "Triplets of (param_name, param_type, values) for tuning. "
                 "See the comments in the source code for details.")
    ap.add_argument('--best_params', '-P',
        help="Tab-separated file containing best parameters for each treebank")
    ap.add_argument('--output', '-O')
    ap.add_argument('--size', '-s', type=int, default=1000, metavar='S',
        help="Sample size. Only a random sample of S word types will be used.")
    ap.add_argument('--ngmax', '-M', type=int, default=5)
    ap.add_argument('--nproc', '-j', type=int, default=1)
    ap.add_argument('--k', '-k', type=int, default=3,
            help="K for K-fold cross validation.")
    ap.add_argument('--max-iter', '-m', type=int, default=5000,
            help="Maximum iterations during random search.")
    ap.add_argument('--feature-type', '-F',
            choices=("sparse","sparse2", "onehot"), default="sparse2")
    ap.add_argument('--cross-features', '-C', type=int, default=2)
    ap.add_argument('--strategy', '-S', choices=("mono","twostep"),
            default="mono")
    args = ap.parse_args()

    if args.command == 'tune':
        params = (('treebank', 'c', args.treebanks),
                  ('C', 'n', (0.01, 100.0, 0.1)),
                  ('window', 'n', (2, 7, 1)),
                 )
        if args.params:
            params = list(eval(args.params))
            params.append(('treebank', 'c', tuple(args.treebanks)))

        pool = Pool(processes=args.nproc)
        fit_func = FitFunction(feature_type=args.feature_type,
                    cross_features=args.cross_features,
                    classifier=args.strategy,
                    data_size=args.size) 
#        fit_func({'C':10.0, 'treebank': args.treebanks[0], 'window': 2})
        pool.map(fit_func, random_iter(params, max_iter=args.max_iter))
    elif args.command in {'test', 'score'}:
        paramstr = None
        params_dict = dict()
        if args.params:
            paramstr = args.params
        if args.best_params:
            with open(args.best_params, 'r') as fp:
                for line in fp:
                    tb, par = line.strip().split()
                    tb = os.path.basename(tb).replace('.conllu', '').replace('UD_', '')
                    params_dict[tb] = par
        for tb in args.treebanks:
            tbname = os.path.basename(tb).replace('.conllu', '').replace('UD_', '')
            if params_dict:
                paramstr = params_dict.get(tbname, paramstr)
            assert(paramstr is not None)
            params = dict()
            for k, v in (kv.split('=') for kv in paramstr.split(',')):
                try:
                    params[k] = int(v)
                except:
                    try:
                        params[k] = float(v)
                    except:
                        params[k] = v
            params['treebank'] = tb
            scores = train_test(params, 
                        k=args.k, max_size=args.size, feature_type=args.feature_type,
                        cross_features=args.cross_features, classifier=args.strategy)
            print("{}\t{}\t{}\n".format(tb, *scores.mean(axis=0).tolist()))
