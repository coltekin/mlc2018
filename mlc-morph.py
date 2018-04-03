#!/usr/bin/env python3
"""
"""

import sys
import os.path
from collections import Counter
import argparse
import numpy as np

from conllu import conllu_sentences


def get_ttr(nodes):
    return len(set([x.form for x in nodes])) / len(nodes)

def get_msp(nodes):
    return (len(set([x.form for x in nodes]))
            / len(set([x.lemma for x in nodes])))

def get_feat_entropy(nodes):
    feat_dist = Counter()
    pos_dist = Counter()
    n_pos, n_feat = 0, 0
    for node in nodes:
        if node.feats:
            feats = node.feats.split('|')
            feat_dist.update(feats)
            n_feat += 1
        pos_dist.update([node.upos])
        n_pos += 1
    ent_pos, ent_feat = 0, 0
    for pos in pos_dist:
        p = pos_dist[pos] / n_pos
        ent_pos -= p * np.log2(p)
    for feat in feat_dist:
        p = feat_dist[feat] / n_feat
        ent_feat -= p * np.log2(p)
    return ent_pos, len(pos_dist), ent_feat, len(feat_dist)

def get_cond_entropy(nodes):
    feat_map = dict()   # mapping from features to forms
    form_map = dict()   # mapping from forms to features

    for node in nodes:
        feat = '\t'.join((str(node.lemma), node.upos, str(node.feats)))
        if feat not in feat_map: feat_map[feat] = Counter()
        feat_map[feat].update([node.form])
        if node.form not in form_map: form_map[node.form] = Counter()
        form_map[node.form].update([feat])
    ent_form_feat = 0    # H(form|feat)
    for form in form_map:
        for feat in form_map[form]:
            joint_p = form_map[form].get(feat) / len(nodes)
            cond_p = form_map[form].get(feat) / sum(form_map[form].values())
            ent_form_feat -= joint_p * np.log2(cond_p)
    #TODO: avoid repetition
    ent_feat_form = 0    # H(feat|form)
    for feat in feat_map:
        for form in feat_map[feat]:
            joint_p = feat_map[feat].get(form) / len(nodes)
            cond_p = feat_map[feat].get(form) / sum(feat_map[feat].values())
            ent_feat_form -= joint_p * np.log2(cond_p)

    return ent_form_feat/len(nodes), ent_feat_form/len(nodes)



def sample(nodes, n, pos_bl=['NUM', 'PUNCT'], pos_wl=[]):
    if pos_wl:
        filtered = [x for x in nodes if x.upos in pos_wl]
    else:
        filtered = [x for x in nodes if x.upos not in pos_bl]
    return np.random.choice(filtered, n)

def parse_cmdline(args):
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--samples', default=10, type=int, help='number of samples')
    ap.add_argument('-S', '--sample-size', default=1000, type=int)
    ap.add_argument('--separator', default='\t')
    ap.add_argument('files', metavar='CoNLL-U file', nargs='+')
    opt = ap.parse_args()
    return opt


opt = parse_cmdline(sys.argv)

head = True
for fname in opt.files:
    nodes = []
    for sent in conllu_sentences(fname):
        nodes.extend(sent.nodes[1:])

    ttr = []
    msp = []
    pos_ent = []
    pos_count = []
    feat_ent = []
    feat_count = []
    cent_form_feat = []
    cent_feat_form = []

    for _ in range(opt.samples):
        smpl = sample(nodes, opt.sample_size)
        ttr.append(get_ttr(smpl))
        msp.append(get_msp(smpl))
        pe, pc, fe, fc = get_feat_entropy(smpl)
        pos_ent.append(pe)
        pos_count.append(pc)
        feat_ent.append(fe)
        feat_count.append(fc)
        form_feat, feat_form = get_cond_entropy(smpl)
        cent_form_feat.append(form_feat)
        cent_feat_form.append(feat_form)

    fmt = "{}" + "{}{{}}".format(opt.separator) *16
    if head:
        print("# sample_size = {}, samples = {}".format(
            opt.sample_size, opt.samples))
        print(fmt.format('fname', 'ttr', 'ttr_sd', 'msp', 'msp_sd',
            'pos_ent', 'pos_ent_sd', 'pos_types', 'pos_types_sd',
            'feat_ent', 'feat_ent_sd', 'feat_types', 'feat_types_sd',
            'cent_form_feat', 'cent_form_feat_sd',
            'cent_feat_form', 'cent_feat_form_sd'))
        head = False
    print(fmt.format(os.path.basename(fname).replace('.conllu', ''),
                     np.mean(ttr), np.std(ttr),
                     np.mean(msp), np.std(msp),
                     np.mean(pos_ent), np.std(pos_ent),
                     np.mean(pos_count), np.std(pos_count),
                     np.mean(feat_ent), np.std(feat_ent),
                     np.mean(feat_count), np.std(feat_count),
                     np.mean(cent_form_feat), np.std(cent_form_feat), 
                     np.mean(cent_feat_form), np.std(cent_feat_form)))
