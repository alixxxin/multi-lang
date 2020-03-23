"""

    Main script for inferring correspondences across domains by using the
    Gromov-Wasserstein barycenter algorithm.

    Parts of the machinery to load / evaluate word embeddings where built upon
    the very thorough codebase by dmelis https://github.com/dmelis/otalign

"""
import sys
import os
import argparse
import collections
from collections import defaultdict
from time import time
import pickle

import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pylab as plt

import pdb
import ot

from src.bilind import *
import src.embeddings as embeddings
import src.multilind as multilind
from src.gw_optim import get_entropic_gromov_barycenters,compute_gromov_wasserstein

def dump_results(outdir, args, optim_args, acc, BLI):
    results = {'acc': acc, 'args': vars(args), 'optim_args':  vars(optim_args)}
    if BLI.mapping is not None:
        results['P'] = BLI.mapping
    np.save(os.path.join(outdir, "coupling"), BLI.coupling)
    dump_file = os.path.join(outdir, "results.pkl")
    pickle.dump(results, open(dump_file, "wb"))

def load_results(outdir, BLI):
    dump_file = os.path.join(outdir, "results.pkl")
    results = pickle.load(open(dump_file, "rb"))
    BLI.mapping  = results['P']
    BLI.coupling = results['G']
    return BLI

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
        return ret_di
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi lingual Word embedding alignment with GW barycenter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    ### General Task Options
    general = parser.add_argument_group('General task options')
    general.add_argument('--debug',action='store_true',
                    help='trigger debugging mode (saving to /tmp/)')
    general.add_argument('--data_dir', type=str, default='data/raw',
                         help='where word embedding data is located (i.e. path to MUSE/data dir)')
    general.add_argument('--load', action='store_true',
                         help='load previously trained model')
    general.add_argument('--task', type=str, default='conneau', choices = ['dinu','xling', 'zhang','conneau'],
                         help='which task to test on')
    general.add_argument('--encoding', type=str,
                         default='utf-8', help='embedding encoding')
    general.add_argument('--maxs', type=int, default=5000,
                         help='use only first k embeddings from source [default: 5000]')
    general.add_argument('--maxt', type=int, default=5000,
                         help='use only first k embeddings from target [default: 5000]')
    general.add_argument('--distribs', type=str, default='uniform',choices=['uniform', 'zipf'],
                         help='p/q distributions to use [default: uniform]')
    general.add_argument('--normalize_vecs', type=str, default='both',
                         choices=['mean','both','whiten','whiten_zca','none'], help='whether to normalize embeddings')
    general.add_argument('--score_type', type=str, default='coupling', choices=[
                       'coupling','transported','distance'], help='what variable to use as the basis for translation scores')
    general.add_argument('--adjust', type=str, default='none', choices=[
                       'csls','isf','none'], help='What type of neighborhood adjustment to use')
    general.add_argument('--maxiter', type=int, default=20, help='Max number of iterations for barycenter calculation')
    general.add_argument('--convergeiter', type=int, default=5, help='Max number of iterations for multilind alignment.')
    general.add_argument('--lang_space', type=str, default='en',
                        help='language space project to')
    general.add_argument('--mapping_dir', type=str, default='none',
                        help='language space matrix project to')

    general.add_argument('--option', type=str, default='barycenter', choices=[
                        'barycenter','gw','unw'], help="what model to use for computing barycenter")
    general.add_argument('--test', type=bool, default=False)
    general.add_argument('--initlang', type=str, default='random',
                        help="what to use for barycenter initialization, random or some language set")
    general.add_argument('--dim', type=str, default='default', choices=['default', 'sum', '2times'],
                        help="dimension of barycenter initialization, default is the size of language set")
     

    


    #### PATHS
    general.add_argument('--chkpt_path', type=str,
                         default='checkpoints', help='where to save the snapshot')
    general.add_argument('--results_path', type=str, default='out',
                         help='where to dump model config and epoch stats')
    general.add_argument('--log_path', type=str, default='log',
                         help='where to dump training logs  epoch stats (and config??)')
    general.add_argument('--summary_path', type=str, default='results/summary.csv',
                         help='where to dump model config and epoch stats')

    ### SAVING AND CHECKPOINTING
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency during train (in iters)')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='checkpoint save frequency during train (in  iters)')
    parser.add_argument('--plot_freq', type=int, default=100,
                        help='plot frequency during train (in  iters)')


    ###  LANGUAGE TREE STRUCT
    general.add_argument('--tree', type=str, default='test-tree',
                         help='which lingual tree structure')

    #############   Gromov-specific Optimization Args ###############
    gromov_optim = parser.add_argument_group('Gromov Wasserstein Optimization options')

    gromov_optim.add_argument('--metric', type=str, default='cosine', choices=[
                           'euclidean', 'sqeuclidean', 'cosine', 'square_loss'], help='metric to use for computing vector distances')
    gromov_optim.add_argument('--normalize_dists', type=str, default='mean', choices = ['mean','max','median','none'],
                       help='method to normalize distance matrices')
    gromov_optim.add_argument('--no_entropy', action='store_false', default=True, dest='entropic',
                       help='do not use entropic regularized Gromov-Wasserstein')
    gromov_optim.add_argument('--entreg', type=float, default=5e-4,
                       help='entopy regularization for sinkhorn')
    gromov_optim.add_argument('--mapping_entreg', type=float, default=5e-4,
                       help='entopy regularization for infer barycenter to node mapping')
    gromov_optim.add_argument('--tol', type=float, default=1e-8,
                       help='stop criterion tolerance for sinkhorn')
    gromov_optim.add_argument('--gpu', action='store_true',
                       help='use CUDA/GPU for sinkhorn computation')
    gromov_optim.add_argument('--gwmaxiter', default=50,
                       help='Max number of iteration for optimizing over gw mapping')
    gromov_optim.add_argument('--bregmanmaxiter', default=25,
                       help='Max number of iteration for bregman projection')
    
    args = parser.parse_args()

    if args.debug:
        args.verbose      = True
        args.chkpt_path   = '/tmp/'
        args.results_path = '/tmp/'
        args.log_path     = '/tmp/'
        args.summary_path = '/tmp/'

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


    optimp = gromov_optim
    optimp.normalize_vecs  = args.normalize_vecs
    optimp.normalize_dists = args.normalize_dists

    optim_args = argparse.Namespace(
        **{a.dest: getattr(args, a.dest, None) for a in optimp._group_actions})
    data_args = argparse.Namespace(
        **{a.dest: getattr(args, a.dest, None) for a in general._group_actions})

    return args, optim_args

def make_path(root, args):
    if root is None:
        return None
    topdir = '_'.join(["multilang", str(args.maxs)])
    method = 'gromov'
    params = {  # Subset of parameters to put in filename
                'entreg': 'ereg',
                'tol': 'tol',
    }
    subdir = [method, args.normalize_vecs, args.metric, args.distribs]
    for arg, name in params.items():
        val = getattr(args, arg)
        subdir.append(params[arg] + '_' + str(val))
    subdir = '_'.join(subdir)
    path = os.path.join(root, args.task, topdir, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def print_header(method):
    print('='*80)
    print('='*13 +'  Bilingual Lexical Induction with Gromov-Wasserstein  ' + ('='*12))
    print('='*80)


def main():
    args, optim_args = parse_args()
    outdir   = make_path(args.results_path, args)
    print('Saving results to: {}'.format(outdir))

    if args.option == "gw":
        # Instantiate Multilingual Lexical Induciton Object
        MLI = multilind.multi_lang_embedding(args, optim_args, outdir, mappingdir=outdir)
        langpairs = MLI.train(MLI.create_evaluation)
        MLI.retrain_mapping_to_child()

        for lang1 in langpairs:
            for lang2 in langpairs:
                if lang1!=lang2:
                    MLI.testing(lang1, lang2)

    if args.option == "unw" or args.option == "barycenter":
        if args.mapping_dir != 'none':
            mapping_dir = args.mapping_dir
        elif args.task == 'conneau' or 'xling':
            mapping_dir = "tempdump/conneau"
        elif args.task == "dinu":
            mapping_dir = "tempdump-dinu/dinu"
        MLI = multilind.barycenter_mapping(args, optim_args, outdir, mappingdir=mapping_dir)
        langpairs = MLI.train()

        if args.task == 'conneau' or args.task == 'xling':
            for lang1 in langpairs:
                for lang2 in langpairs:
                    if lang1!=lang2:
                        MLI.testing(lang1, lang2)
        else:
            MLI.testing('en','fi')
            MLI.testing('en','es')
            MLI.testing('en','it')
            MLI.testing('en','de')

    print('Done!')


if __name__ == "__main__":
    main()

