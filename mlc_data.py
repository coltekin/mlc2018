#!/usr/bin/env python3

import sys, glob, argparse, os, re, random
from udtools.conllu import conllu_sentences

def to_lower(s, lang):
    """Workaround Turkish dottless-i.
    """
    if lang in {'tr', 'tur', 'Turkish'}:
        s = s.replace('İ', 'i').replace('I', 'ı')
    return s.lower()

lang_re = re.compile(r'.*UD_([A-Za-z_]+)-.*')
def read_treebank(tb, max_size=None,
        shuffle=False, lowercase=True, skip_multi=True,
        pos_filter={'NOUN', 'VERB', 'ADJ', 'ADV'}):
    d = set()
    n_sent, n_node = 0, 0
    m = lang_re.match(tb)
    lang = m.group(1)
    for tbf in glob.glob(tb + '/*.conllu'):
        for sent in conllu_sentences(tbf):
            n_sent += 1
            for node in sent.nodes:
                if not (node.form and node.lemma) \
                        or sent.get_multi(node) \
                        or node.upos not in pos_filter:
                    continue
                n_node += 1
                feats = [node.upos]
                if node.feats is not None:
                    feats += node.feats.split("|")
                feats = tuple(feats)
                if lowercase:
                    d.add((to_lower(node.form, lang),
                        to_lower(node.lemma, lang),
                        feats))
                else:
                    d.add((node.form, node.lemma, feats))
    d = list(d)
    if shuffle: random.shuffle(d)
    if max_size: d = d[:max_size]
    return d

class Node:
    __slots__ = ('form', 'lemma', 'upos')
    def __init__(self, form=None, lemma=None, upos=None):
        self.form = form
        self.lemma = lemma
        self.upos = upos

def read_pbc(filename, max_size=None, shuffle=False, lowercase=True,
        form_filter=r'[^.?!“”0-9]+'):
    if form_filter:
        filter_re = re.compile(form_filter)
    lcode = filename.split('-',1)[0]
    nodes = []
    with open(filename, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('# language_name:'):
                lang = line.strip().split()[-1]
            elif '\t' in line and not line.startswith('#'):
                id_, tokens = line.strip().split('\t')
                tokens = tokens.lower().split()
                if form_filter:
                    tokens = [Node(t) for t in tokens if filter_re.match(t)]
                else:
                    tokens = [Node(t) for t in tokens]
                nodes.extend(tokens)
    return nodes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('input_files', nargs="+")
    args = ap.parse_args()
    
    for fn in args.input_files:
        nodes = read_pbc(fn, form_filter=None)
        print(fn, len(nodes))
