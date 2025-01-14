import torch
import torch.nn as nn

# from multiprocessing import Pool
from mpire import WorkerPool

import math, random, sys
from optparse import OptionParser
import pickle

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(os.getcwd())

from fast_jtnn import *
import rdkit

def tensorize(smiles, assm=True):
    if not smiles: return ''
    mol_tree = MolTree(smiles)
    if (mol_tree.n_errors > 0):
        return ''
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    num_splits = int(opts.nsplits)

    data = []
    with open(opts.train_path) as f:
        for line in f:
            line = line.strip("\r\n ")
            if not line:
                continue
            data.append(line.split()[0])

    # print('Read all data!')
    # pool = Pool(opts.njobs)
    # all_data = pool.map(tensorize, data)a

    # mpire library provides better multiprocessing support.
    # Reports error with full stacktrace and supports excellent progress bar.

    with WorkerPool(n_jobs=opts.njobs) as pool:
        results = pool.map(tensorize, data, progress_bar=True)

    total_input_count = len(results)
    print('The total count of preprocess input entries: ', total_input_count)

    all_data = list(filter(lambda x: (not not x), results))
    print('Input count after filtering out invalid entries: ', len(all_data))

    # Write filtered training set to output
    # with open('/tmp/valid_train.txt', 'w') as file:
    #   out_buf = '\n'.join(all_data)
    #   file.write(out_buf)
    # print('Finished writing out_buf: ', out_buf)

    le = int((len(all_data) + num_splits - 1) / num_splits)

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    print('Finished preprocess. Check out the tensors-* files outputs.')
