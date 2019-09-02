#!/usr/bin/env python3

import sys, collections
import numpy as np
import pandas as pd
from align import simple_align
from difflib import SequenceMatcher

def align_lcs(s1, s2):
    s = SequenceMatcher(None, s1, s2)
    m = s.find_longest_match(0, len(s1), 0, len(s2))
    matched = list(s1[m.a:m.a+m.size])
    s1pfx, s2pfx = list(s1[:m.a]), list(s2[:m.b])
    s1sfx, s2sfx = list(s1[m.a+m.size:]), list(s2[m.b+m.size:])
    if len(s1pfx) < len(s2pfx):
        s1pfx = (len(s2pfx) - len(s1pfx)) * [''] + s1pfx 
    if len(s2pfx) < len(s1pfx):
        s2pfx = (len(s1pfx) - len(s2pfx)) * [''] + s2pfx 

    if len(s1sfx) < len(s2sfx):
        s1sfx += (len(s2sfx) - len(s1sfx)) * ['']
    if len(s2sfx) < len(s1sfx):
        s2sfx += (len(s1sfx) - len(s2sfx)) * ['']

    return list(zip(s1pfx + matched + s1sfx, s2pfx + matched + s2sfx))

def simple_scorer(s, t):
    return 0 if s == t else 1

def weighted_scorer(s, t, wm):
    try:
        return wm[s, t]
    except:
        return simple_scorer(s, t)

def count_gaps(seq):
    gaps = 0
    prev = None
    for s in seq:
        if s == '' and prev not in {'', '<'}:
            gaps += 1
        if s == '>' and prev == '':
            gaps -= 1
        prev = s
    return gaps

def print_alignment(align):
    for pair in align:
        print("{:<2}".format(pair[0] or "-"), end="")
    print()
    for pair in align:
        print("{:<2}".format(pair[1] or "-"), end="")
    print()

class Aligner:
    def __init__(self, method='lcs', source_chars=None, target_chars=None):
        self.scorer = simple_scorer
        self.source_chars = source_chars
        self.target_chars = target_chars
        self.method = method

    def fit(self, source, target):
        if self.source_chars is None or self.target_chars is None:
            self.source_chars, self.target_chars = {''}, {''}
            for s, t in zip(source, target):
                self.source_chars.update(s)
                self.target_chars.update(t)
        self.weights = pd.DataFrame(1, index=sorted(self.source_chars),
                                       columns=sorted(self.target_chars))
        for i in self.source_chars:
            for j in self.target_chars:
                if i and j and i == j:
                    self.weights.loc[i,j] = 0.0
                if i == '':
                    self.weights.loc[i,j] = 100.0
                if j == '':
                    self.weights.loc[i,j] = 1.0
        counts = pd.DataFrame(0, index=sorted(self.source_chars),
                                 columns=sorted(self.target_chars))

        self.scorer = lambda x,y: weighted_scorer(x, y, self.weights.loc)
        score_sum = 0
        print(self.weights)
        for l, w in zip(lemmas, words):
            for align, sc in a.align(l, w, return_score=True):
                print_alignment(align)
                print(sc)
                for s, t in align:
                    counts.loc[s, t] += 1
                score_sum += sc
        print('=======', score_sum)
        total = counts.sum().sum()
        rowsum = counts.sum(axis=1)
        colsum = counts.sum(axis=0)
        self.weights = total * counts.div(rowsum, axis=0).fillna(0)
        self.weights = self.weights.div(colsum, axis=1).fillna(0)
        self.weights = 1 / (1 + self.weights)
        print(self.weights)

        for l, w in zip(lemmas, words):
            for align, sc in a.align(l, w, return_score=True):
#                print_alignment(align)
#                print(sc)
                for s, t in align:
                    counts.loc[s, t] += 1
                score_sum += sc
        print('=======', score_sum)

    def align(self, s, t, gap_penalty=2, return_score=False):
        if self.method == 'lcs':
            yield align_lcs(s, t)
            return
        alignments = list(simple_align(s, t, self.scorer))
        penalty = []
        for a in alignments:
            s_gaps = count_gaps([x[0] for x in a.corr])
            t_gaps = count_gaps([x[1] for x in a.corr])
            print_alignment(a.corr)
            print('---- gaps:', s_gaps, t_gaps, 'delta:', a.delta )
            penalty.append(a.delta + gap_penalty*(s_gaps + t_gaps))
        penalty = np.array(penalty)
        for i in np.where(penalty == penalty.min())[0]:
            a = alignments[i]
#            print('+++++++++++++++++++++++++')
#            print_alignment(a.corr)
#            print('+++++++++++++++++++++++++')
            if return_score:
                yield a.corr, a.delta
            else:
                yield a.corr

if __name__ == "__main__":
    a = Aligner(method='lcs')
    algn = next(a.align("<orfan>", "<definite articulation>"))
    print_alignment(algn)
    algn = next(a.align("<definite articulation>", "<orfan>"))
    print_alignment(algn)
    algn = next(a.align("gewordenabcde","werden"))
    print_alignment(algn)
    lemmas, words = [], []
    with open(sys.argv[1], 'r') as fp:
        for line in fp:
            l , w, _ = line.strip().split('\t')
            lemmas.append('<'+ l + '>')
            words.append('<' + w + '>')
            align = next(a.align(l, w))
            print_alignment(align)
            align = next(a.align(w, l))
            print_alignment(align)
            print()

#    for l, w in zip(lemmas, words):
#        print("-----")
#        for align in a.align('<'+l+'>', '<'+w+'>'):
#            for s, t in align:
#                print(s, '-', t)
#            print()
