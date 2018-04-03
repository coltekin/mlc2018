#!/usr/bin/env python3

import sys
from collections import Counter
from conllu import conllu_sentences

pos_dist = Counter()
wlen_dist = Counter()
rel_dist = Counter()
feat_dist = dict()

n_sent = 0
n_token = 0
n_multi = 0
n_empty = 0
n_lemma = 0
n_feat = 0

lemma_type = set()
form_type = set()

for sent in conllu_sentences(sys.argv[1]):
    n_sent += 1
    n_token += len(sent.nodes) - 1
    n_multi += len(sent.multi)
    n_empty += len(sent.empty)
    for node in sent.nodes[1:]:
        pos_dist.update([node.upos])
        rel_dist.update([node.deprel])
        form_type.add(node.form)
        n_lemma += int(node.lemma != None)
        lemma_type.add(node.lemma)
        n_feat += int(node.feats != None)
        if node.feats:
            for f in node.feats.split('|'):
                feat, val = f.split('=')
                if feat not in feat_dist:
                    feat_dist[feat] = Counter()
                feat_dist[feat].update([val])


# print("Sentences: {}".format(n_sent))
# print("Tokens: {}".format(n_token))
# print("Multi: {}".format(n_multi))
# print("Empty: {}".format(n_empty))
# print("Lemma: {}".format(n_lemma))
# print("Feat: {}".format(n_feat))
# print()
# print("Form types: {}".format(len(form_type)))
# print("Lemma types: {}".format(len(lemma_type)))
# 
# print(pos_dist)
# print(rel_dist)
# for feat in feat_dist:
#     print(feat, feat_dist[feat])

fmt = "{}" + "\t{}" * 11
print(fmt.format("", "n_sent", "n_tok",  "n_lemma", "n_feat", "n_multi",
    "n_empty", "form_type", "lemma_type", "pos_type", "rel_type",
    "feat_type"))
print(fmt.format(sys.argv[1], n_sent, n_token,  n_lemma, n_feat, n_multi,
    n_empty, len(form_type), len(lemma_type), len(pos_dist), len(rel_dist),
    len(feat_dist)))

print(pos_dist)

