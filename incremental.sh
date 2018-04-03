#!/bin/bash
for i in $(seq 1 40);do 
	n=$((2500 * $i)) ;
	./mlc-morph.py -s 1000 -S $n data/UD_*/*.conllu > results/incremental-$n &
done
