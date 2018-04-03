#!/usr/bin/env python3 

from collections import defaultdict
import glob, sys
import numpy as np


for fname in sorted(glob.glob("Data/UD_*/*.conllu")):
    uni_gram_freq = defaultdict(float)
    bi_gram_freq = defaultdict(float)
    log_prob = defaultdict(float)
    sent = ["<s>"]

    for line in open(fname, "r"):
        if line.startswith("#"):
            continue
        elif len(line.strip()) == 0:
            sent.append("</s>")
            sent.append("<s>")
        else:
            sent.append(line.split("\t")[3])
    #print(sent[:10])
    for w1, w2 in zip(sent[:-1], sent[1:]):
        if w1 == "</s>" or w2 == "<s>":
            continue
        bi_gram_freq[w1,w2] += 1.0
        uni_gram_freq[w1] += 1.0

    for k, v in bi_gram_freq.items():
        log_prob[k[1],k[0]] = np.log2(bi_gram_freq[k[0],k[1]]) - np.log2(uni_gram_freq[k[0]])

    perplexity, N, temp_perp, S = 0.0, 0.0, 0.0, 0.0
    
    for w1, w2 in zip(sent[:-1], sent[1:]):
        if w1 == "</s>" or w2 == "<s>":
            temp_perp = (-1.0*temp_perp)/N
            #print(np.power(2.0, temp_perp))
            perplexity += np.power(2.0, temp_perp)
            temp_perp, N = 0.0, 0.0
            S += 1.0
            continue
        else:
            temp_perp += log_prob[w2,w1]
            N += 1.0

    print(fname, perplexity, S, round(perplexity/S,4), np.log2(perplexity/S), sep="\t")
            
