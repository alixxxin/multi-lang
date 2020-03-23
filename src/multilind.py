import gc
import os
import pdb
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy as sp
import scipy.linalg
from ot.utils import dist
from scipy.linalg import qr
from scipy.stats import describe
from six.moves import cPickle as pickle 

import src.embeddings as embeddings
from src.gw_optim import get_entropic_gromov_barycenters,compute_gromov_wasserstein
import gc
import src.bilind as bilind
from pulp import *
import argparse

from src.barycenter_optim import cal_barycenter_for, build_MXY, unweighted_barycenter
from src import orth_procrustes


try: 
    from ot.gpu import bregman
    from gw_optim_gpu import gromov_wass_solver
except ImportError:
    from src.gw_optim import gromov_wass_solver

plt.switch_backend('agg')

def load_vectors(args):
    """
        Assumes file structure as in MUSE repo.
    """
    dict_fold = 'train' # which fold of the data will be used to produce results
    if args.task == 'conneau' or 'xling':
        data_dir = os.path.join(args.data_dir, 'MUSE')
        dict_dir = os.path.join(data_dir, 'crosslingual/')
        if args.task == 'xling':
            dict_dir = os.path.join(dict_dir, 'xling-dictionaries/bli_datasets/')
        else:
            dict_dir = os.path.join(dict_dir, 'dictionaries/')

        src_path = os.path.join(data_dir, 'wiki.' + args.src_lang + '.vec')
        trg_path = os.path.join(data_dir, 'wiki.' + args.trg_lang + '.vec')
        src_freq_path = None
        trg_freq_path = None
        if dict_fold == 'test':
            postfix = '.5000-6500.txt'
        elif dict_fold == 'train':
            postfix = '.0-5000.txt'
        else:
            raise ValueError('Unrecognized dictionary fold for evaluation')
    elif args.task == 'dinu':
        data_dir = os.path.join(args.data_dir,'dinu')
        dict_dir = os.path.join(data_dir, 'dictionaries/')
        src_path = os.path.join(data_dir, 'embeddings', args.src_lang + '.emb.txt')
        trg_path = os.path.join(data_dir, 'embeddings', args.trg_lang + '.emb.txt')
        src_freq_path = None
        trg_freq_path = None
        postfix  = '.{}.txt'.format(dict_fold)
    elif args.task == 'zhang':
        order = [args.src_lang,args.trg_lang]
        if args.src_lang == 'en':
            order = order[::-1]
        data_dir = os.path.join(args.home_dir,'pkg/UBiLexAT/data/','-'.join(order))
        dict_dir = data_dir
        src_path = os.path.join(data_dir, 'word2vec.' + args.src_lang)
        trg_path = os.path.join(data_dir, 'word2vec.' + args.trg_lang)
        src_freq_path = os.path.join(data_dir, 'vocab-freq.' + args.src_lang)
        trg_freq_path = os.path.join(data_dir, 'vocab-freq.' + args.trg_lang)
        postfix  = '.train.txt'

    srcfile = open(src_path, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(trg_path, encoding=args.encoding, errors='surrogateescape')
    src_words, xs = embeddings.read(srcfile, args.maxs)
    trg_words, xt = embeddings.read(trgfile, args.maxt)
    srcfile.close()
    trgfile.close()
    
    if src_freq_path:
        with open(src_freq_path, encoding=args.encoding, errors='surrogateescape') as f:
            lines = [a.split(' ') for a in f.read().strip().split('\n')]
            freq_src = {k: int(v) for (k,v) in lines}

        with open(trg_freq_path, encoding=args.encoding, errors='surrogateescape') as f:
            lines = [a.split(' ') for a in f.read().strip().split('\n')]
            freq_trg = {k: int(v) for (k,v) in lines}

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    if args.task == 'zhang':
        dict_path = os.path.join(dict_dir,  'all.' + '-'.join(order) + '.lex')
        flip = False
    elif args.task == 'dinu' and args.src_lang != 'en':
        # Only has dicts in one direction, flip
        dict_path = os.path.join(dict_dir, args.trg_lang + '-' + args.src_lang + postfix)
        src_to_en = os.path.join(dict_dir, 'en' + '-' + args.src_lang + postfix)
        en_to_trg = os.path.join(dict_dir, args.trg_lang + '-' + 'en' + postfix)
        flip = True
    elif args.task == 'xling':
        dict_path = os.path.join(dict_dir, args.src_lang+'-'+args.trg_lang+'/yacle.test.freq.2k.'+args.src_lang+'-' + args.trg_lang + '.tsv')
        src_to_en = os.path.join(dict_dir, args.src_lang+'-'+'en'+'/yacle.test.freq.2k.'+args.src_lang+'-' + 'en' + '.tsv')
        en_to_trg = os.path.join(dict_dir, 'en'+'-'+args.trg_lang+'/yacle.test.freq.2k.'+'en'+'-' + args.trg_lang + '.tsv')

        flip = False
        if not os.path.exists(dict_path):
            dict_path = os.path.join(dict_dir, args.trg_lang+'-'+args.src_lang+'/yacle.test.freq.2k.'+args.src_lang+'-' + args.trg_lang + '.tsv')
            flip = True

    else:
        src_to_en = os.path.join(dict_dir, args.src_lang + '-' + 'en' + postfix)
        en_to_trg = os.path.join(dict_dir, 'en' + '-' + args.trg_lang + postfix)
        dict_path = os.path.join(dict_dir, args.src_lang + '-' + args.trg_lang + postfix)
        flip = False


    if not os.path.exists(dict_path):
        # create new dict
        print('Warning: no dict found, creating dictionary')
        create_dict_for(src_to_en, en_to_trg, dict_path, args)

    dictf = open(dict_path, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    max_srcind = 0 # These are mostly for debug
    max_trgind = 0
    for line in dictf:
        splitted = line.split()
        if len(splitted) > 2:
            # Only using first translation if many are provided
            src, trg = splitted[:2]
        elif len(splitted) == 2:
            src, trg = splitted
        else:
            # No translation? Only happens for Zhang data so far
            continue
        if flip: src, trg = trg, src
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
            max_srcind = max(max_srcind, src_ind)
            max_trgind = max(max_trgind, trg_ind)
        except KeyError:
            oov.add(src)

    return xs, xt, src_words, trg_words,  src_word2ind, trg_word2ind, src2trg


def create_dict_for(src_to_en, en_to_trg, dict_path, args):
    en_trg_dict = {}
    en_trg = open(en_to_trg, encoding=args.encoding, errors='surrogateescape')

    for line in en_trg:
        splitted = line.replace('\n','').split()
        if len(splitted)>2:
            en, trg = splitted[0], splitted[1:]
        elif len(splitted) == 2:
            en, trg = splitted[0], [ splitted[1] ]
        else:
            continue

        if en in en_trg_dict.keys():
            en_trg_dict[en].extend(trg)
        else:
            en_trg_dict[en] = trg
    en_trg.close()
    src_en = open(src_to_en, encoding=args.encoding, errors='surrogateescape')
    dictf = open(dict_path, "w", encoding=args.encoding)
    for line in src_en:
        splitted = line.replace('\n','').split()
        src, en = splitted[0], splitted[1:]
        for en_word in en:
            if en_word in en_trg_dict.keys():
                for trg in en_trg_dict[en_word]:
                    dictf.write(src+'\t'+trg+'\n')
    src_en.close()
    dictf.close()

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

class LanguageTree():
    barycenter = 1
    def __init__(self, is_barycenter, child, lang=None, parent=None, child_num=0, lbdas=None):
        self.is_barycenter = is_barycenter
        self.barycenter_mtrx = None
        self.child = child
        self.child_num = child_num
        self.freq = None
        self.mapping_to_child = None
        self.parent = None
        if lbdas is not None:
            self.lbdas = np.array(lbdas)
        else:
            self.lbdas = lbdas
        if lang is not None:
            self.lang = lang
            self.child_num += 1
        if self.is_barycenter:
            if lang is None:
                self.lang = str(LanguageTree.barycenter)
                LanguageTree.barycenter += 1
            self.child_num += np.sum(list(map(lambda x: x.child_num, child)))
            for i in range(len(child)):
                child[i].parent = self
                child[i].idx = i

    def load_barycenter_mtrx(self, path, mlt):
        if self.is_barycenter:
            # load barycenter form file
            barycenter_postfix = "barycenter_"+str(self.lang)
            filepath = os.path.join(path, barycenter_postfix)
            if os.path.exists(filepath):
                d = load_dict(filepath)
                self.mapping_to_child = d['mapping_to_child']
                self.barycenter_mtrx = d['barycenter_mtrx']
                self.freq = d['freq']
                return self
 
        if not self.lang.isdigit():
            self.barycenter_mtrx = mlt.read_embedding(self.lang)
            if mlt.args.distribs == 'zipf':
                self.freq = zipf_init(self.lang, len(self.barycenter_mtrx))
            else:
                self.freq = ot.unif(len(self.barycenter_mtrx))
        
            return self

    def save_barycenter_mtrx(self, path):
        if not self.is_barycenter:
            return
        barycenter_postfix = "barycenter_"+str(self.lang)
        filepath = os.path.join(path, barycenter_postfix)
        print("Saving barycenter matrix"+str(self.barycenter_mtrx))
        d = {
            'mapping_to_child': self.mapping_to_child,
            'barycenter_mtrx': self.barycenter_mtrx,
            'freq': self.freq
        }
        save_dict(d, filepath)


def full_hier_tree():
    ru = LanguageTree(False, [], 'ru')
    pl = LanguageTree(False, [], 'pl')
    cs = LanguageTree(False, [], 'cs')
    el = LanguageTree(False, [], 'el')
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    pt = LanguageTree(False, [], 'pt')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    en = LanguageTree(False, [], 'en')
    A = LanguageTree(True, [ru, pl, cs])
    B = LanguageTree(True, [it, es, pt, fr])
    C = LanguageTree(True, [de, en])
    D = LanguageTree(True, [A, el, B, C])
    compute_seq = [A, B, C, D]
    lang_dict = {'ru': ru, 'it': it, 'es': es, 'fr': fr, 'de': de, 'en': en, 
            'pl': pl, 'cs': cs, 'el': el, 'pt': pt, str(A.lang): A, str(B.lang): B,
            str(C.lang): C, str(D.lang): D}
    return D, lang_dict, compute_seq


def en_tree():
    ru = LanguageTree(False, [], 'ru')
    pl = LanguageTree(False, [], 'pl')
    cs = LanguageTree(False, [], 'cs')
    el = LanguageTree(False, [], 'el')
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    pt = LanguageTree(False, [], 'pt')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    A = LanguageTree(True, 
                [ru,
                 pl,
                 cs,
                 el,
                 it,
                 es,
                 pt,
                 fr,
                 de], 'en')
    compute_seq = [A]
    lang_dict = {'ru': ru, 'it': it, 'es': es, 'fr': fr, 'de': de, 'en': A, 
            'pl': pl, 'cs': cs, 'el': el, 'pt': pt}
    return A, lang_dict, compute_seq


def full_tree():
    ru = LanguageTree(False, [], 'ru')
    pl = LanguageTree(False, [], 'pl')
    cs = LanguageTree(False, [], 'cs')
    el = LanguageTree(False, [], 'el')
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    pt = LanguageTree(False, [], 'pt')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    en = LanguageTree(False, [], 'en')
    A = LanguageTree(True, 
                [ru,
                 pl,
                 cs,
                 el,
                 it,
                 es,
                 pt,
                 fr,
                 de,
                 en])
    compute_seq = [A]
    lang_dict = {'ru': ru, 'it': it, 'es': es, 'fr': fr, 'de': de, 'en': en, 
            'pl': pl, 'cs': cs, 'el': el, 'pt': pt, str(A.lang): A}
    return A, lang_dict, compute_seq


def test_tree2():
    it = LanguageTree(False, [], 'it')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    ru = LanguageTree(False, [], 'ru')
    en = LanguageTree(False, [], 'en')
    hr = LanguageTree(False, [], 'hr')
    A = LanguageTree(True, [
                fr,
                it,
                de,
                en,
                ru,
                hr
                ],
                )

    compute_seq = [A]
    lang_dict = {'it': it, 'fr': fr, 'hr': hr, 'en': en , 'ru': ru, 'de': de, str(A.lang): A}
    return A, lang_dict, compute_seq


def test_tree():
    it = LanguageTree(False, [], 'it')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    ru = LanguageTree(False, [], 'ru')
    en = LanguageTree(False, [], 'en')
    fi = LanguageTree(False, [], 'fi')
    hr = LanguageTree(False, [], 'hr')
    tr = LanguageTree(False, [], 'tr')
    A = LanguageTree(True, [
                fr,
                it,
                de,
                en,
                ru,
                fi,
                hr,
                tr],
                )

    compute_seq = [A]
    lang_dict = {'it': it, 'fi': fi, "tr": tr, 'fr': fr, 'hr': hr, 'en': en , 'ru': ru, 'de': de, str(A.lang): A}
    return A, lang_dict, compute_seq



def s_tree2():
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    en = LanguageTree(False, [], 'en')
    pt = LanguageTree(False, [], 'pt')
    A = LanguageTree(True, [
                fr,
                it,
                es,
                de,
                en,
                pt]
                )
    compute_seq = [A]
    lang_dict = {'it': it, 'es': es, 'fr': fr, 'pt':pt, 'en': en , 'de': de, str(A.lang): A}
    return A, lang_dict, compute_seq



def s_tree():
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    ru = LanguageTree(False, [], 'ru')
    en = LanguageTree(False, [], 'en')
    pt = LanguageTree(False, [], 'pt')
    A = LanguageTree(True, [
                fr,
                it,
                es,
                de,
                en,
                ru,
                pt]
                )

    compute_seq = [A]
    lang_dict = {'it': it, 'es': es, 'fr': fr, 'pt':pt, 'en': en , 'ru': ru, 'de': de, str(A.lang): A}
    return A, lang_dict, compute_seq


def s_hier_tree():
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    ru = LanguageTree(False, [], 'ru')
    fr = LanguageTree(False, [], 'fr')
    de = LanguageTree(False, [], 'de')
    en = LanguageTree(False, [], 'en')
    pt = LanguageTree(False, [], 'pt')
    A = LanguageTree(True, [
                it,
                es,
                pt,
                fr])
    B = LanguageTree(True, [
                en,
                A,
                de
                ])
    compute_seq = [A, B, ru]
    lang_dict = {'pt': pt, 'it': it, 'es': es, 'fr': fr, 'de': de, 'en': en, 'ru': ru, str(A.lang): A, str(B.lang): B }
    return B, lang_dict, compute_seq


def dinu_tree():
    it = LanguageTree(False, [], 'it')
    es = LanguageTree(False, [], 'es')
    fr = LanguageTree(False, [], 'fr')
    fi = LanguageTree(False, [], 'fi')
    en = LanguageTree(False, [], 'en')
    A = LanguageTree(True, [
                it,
                es,
                pt,
                en,
                fr])
    compute_seq = [A]
    lang_dict = {'it': it, 'es': es, 'fr': fr, 'fi': fi, 'en': en, str(A.lang): A }
    return A, lang_dict, compute_seq


class multi_lang_embedding:
    def __init__(self, args, gw_args, outdir):
        self.args = args
        self.gw_args = gw_args
        self.outdir = outdir
        try:
            name = args.tree.replace("-","_")
            self.tree_root, self.lang_dict, self.barycenter_in_seq = globals()[name]()
        except:
            raise NotImplemented(
                "Please choose in ['test-tree','full','selective']")

    def train(self, test_fun):
        langs = self.lang_dict.keys()
        langs = list(filter(lambda x: not x.isdigit(), langs))
        for cnt in range(self.args.convergeiter):
            print("========== iteration "+str(cnt)+" ==========")
            for barycenter in self.barycenter_in_seq:
                # calculate space proj
                self.train2(test_fun)
                # calcluate the alignment
                self.cal_alignment()
        return langs

    def get_mapping(self, coupling, xs, xt):
        best_match_src = coupling.argmax(1) # Best match for each source word
        best_match_trg = coupling.argmax(0) # Best match for each source word

        ns = len(coupling)
        paired = []
        for i in range(ns):
            m = best_match_src[i]
            if best_match_trg[m] == i:
                paired.append((i,m))

        idx_src = [ws for ws,_ in pseudo]
        idx_trg = [wt for _,wt in pseudo]

        xs_nn = xs[idx_src]
        xt_nn = xt[idx_trg]
        P = orth_procrustes(xs_nn, xt_nn)
        return P


    def cal_alignment(self):
        # get mapping
        lang2 = self.args.lang_space
        for lang1 in self.lang_dict.keys():
            if lang1 != lang2 and (not lang1.isdigit()):
                node1 = self.lang_dict[lang1]
                node1.load_barycenter_mtrx(self.outdir, self)
                node2 = self.lang_dict[lang2]
                node2.load_barycenter_mtrx(self.outdir, self)

                # get coupling matrix for lang1 to lang2
                coupling = self.get_lang_mapping(lang1, lang2, self.gw_args.metric, self.gw_args.entreg)                
                # use Procrustes matching to get alignment matrix
                P = self.get_mapping(coupling, node1.barycenter_mtrx, node2.barycenter_mtrx)

                # save projection matrix
                dirname = lang1 + "_" + lang2 + "_5000"
                project_mtrx_path = os.path.join(self.outdir, dirname)
                if not os.path.exists(project_mtrx_path):
                    os.makedirs(project_mtrx_path)
                project_mtrx_path = os.path.join(project_mtrx_path, "mapping.npy")
                np.save(project_mtrx_path, P)


    def train2(self, test_fun):
        for barycenter in self.barycenter_in_seq:
            self.train_barycenter(barycenter, test_fun)

    def retrain_mapping_to_child(self):
        for barycenter in self.barycenter_in_seq:
            barycenter.load_barycenter_mtrx(self.outdir, self)
            C1 = barycenter.barycenter_mtrx
            C1 = np.asarray(C1, dtype=np.float64)
            for i in range(len(barycenter.child)):
                barycenter.child[i].load_barycenter_mtrx(self.outdir, self)
                C2 = barycenter.child[i].barycenter_mtrx
                C2 = np.asarray(C2, dtype=np.float64)
                mapping = compute_gromov_wasserstein(C2, C1, ot.unif(len(C2)), ot.unif(len(C1)), self.gw_args.tol, False, True, self.gw_args.mapping_entreg, self.gw_args.gwmaxiter)
                barycenter.mapping_to_child[i] = mapping
            barycenter.save_barycenter_mtrx(self.outdir)


    def train_barycenter(self, barycenter, test_fun):
        print("training barycenter" + str(barycenter.lang))
        # continue training
        for i in range(len(barycenter.child)):
            barycenter.child[i].load_barycenter_mtrx(self.outdir, self)

        barycenter.load_barycenter_mtrx(self.outdir, self)

        if barycenter.lang.isdigit():
            Cs = [c.barycenter_mtrx for c in barycenter.child]
            ls = [c.lang for c in barycenter.child]
            ps = [ot.unif(len(c.barycenter_mtrx)) for c in barycenter.child]

            # continue iterating to find gromov wasserstein barycenter
            lambdas = [c.child_num for c in barycenter.child]
            lambdas = lambdas / np.sum(lambdas)
            #lambdas = ot.unif(len(Cs))

            barycenter.barycenter_mtrx, barycenter.mapping_to_child = get_entropic_gromov_barycenters(self, test_fun, self.get_lang_mapping, ls, Cs[0], Cs, ps, ot.unif(len(Cs[0])), lambdas, self.gw_args.entreg, self.args.maxiter, self.gw_args.gwmaxiter, self.gw_args.mapping_entreg, tol=self.gw_args.tol, verbose=False, log=False, init_C=barycenter.barycenter_mtrx, loss_fun='square_loss')

            assert(barycenter.barycenter_mtrx[0][0] != None and barycenter.barycenter_mtrx[0][0] != np.nan)
            
            barycenter.save_barycenter_mtrx(self.outdir)

        else:
            # get direct mapping
            barycenter.mapping_to_child = []
            C1 = barycenter.barycenter_mtrx
            q = ot.unif(len(C1))
            C1 = np.asarray(C1, dtype=np.float64)
            for i in range(len(barycenter.child)):
                C2 = barycenter.child[i].barycenter_mtrx
                C2 = np.asarray(C2, dtype=np.float64)
                p = ot.unif(len(C2))
                maping = compute_gromov_wasserstein(C2, C1, q, p, self.gw_args.tol, self.gw_args.gpu, True, self.gw_args.entreg, self.args.maxiter, loss_fun='square_loss')

                barycenter.mapping_to_child.append(maping)
            barycenter.save_barycenter_mtrx(self.outdir)


    def get_lang_mapping(self, lang1, lang2):
        path = self._get_shortest_path_from_lang1_to_lang2(lang1, lang2)
        print(path)
        print(self.lang_dict)

        self.lang_dict[str(path[0][0])].load_barycenter_mtrx(self.outdir, self)

        print(self.lang_dict[str(path[0][0])].mapping_to_child)
        mapping = self.lang_dict[str(path[0][0])].mapping_to_child[path[0][1]]
        print(str(path[0][0])+" to "+str(path[0][1])+'th child', end='')
        if not path[0][2]:
            mapping = np.transpose(mapping)
            print("transpose *", end='')
        for i in range(1, len(path)):
            self.lang_dict[str(path[i][0])].load_barycenter_mtrx(self.outdir, self)
            print("(" +str(path[i][0]) + " to " + str(path[i][1]) + "th child"+ ")", end='')
            if path[i][2]:
                mapping = np.matmul(mapping, self.lang_dict[str(path[i][0])].mapping_to_child[path[i][1]])
            else:
                mapping = np.matmul(mapping, np.transpose(self.lang_dict[str(path[i][0])].mapping_to_child[path[i][1]]))
                print(".T", end='')
            print("*",end='')
        return mapping

    def create_evaluation(self, lang1, lang2, coupling):
        args = self.args
        optim_args = self.gw_args
        outdir = self.outdir
        args.src_lang = lang1
        args.trg_lang = lang2
        xs, xt, src_words, trg_words, src_word2ind, trg_word2ind, src2trg = \
        load_vectors(args)

        BLI = bilind.gromov_bilind(xs, xt, lang1, lang2, src_words, trg_words,
                        src_word2ind, trg_word2ind, src2trg,
                        metric = args.metric, normalize_vecs = args.normalize_vecs,
                        normalize_dists = args.normalize_dists,
                        score_type = args.score_type, adjust = args.adjust,
                        distribs = args.distribs)
        BLI.init_optimizer(**vars(optim_args)) # FIXME: This is ugly. Get rid of it
        print('Fitting bilingual mapping with Gromov Wasserstein')

        # 0. Pre-processing
        BLI.normalize_embeddings()

        BLI.coupling = coupling
        BLI.mapping = BLI.get_mapping()

        # 3. From Couplings to Translation Score
        print('Computing translation scores...')
        print(args.score_type, args.adjust)
        BLI.compute_scores(args.score_type, adjust = args.adjust)

        print('Done!')

        BLI.mapping = BLI.get_mapping(anchor_method = 'mutual_nn', max_anchors = None)

        acc_file = os.path.join(outdir, 'accuracies.tsv')
        acc_dict = {}
        print('Results on test dictionary for fitting vectors: (via coupling)')
        acc_dict['coupling'] = BLI.test_accuracy(verbose=True, score_type = 'coupling')
        print('Results on test dictionary for fitting vectors: (via coupling + csls)')
        acc_dict['coupling_csls'] = BLI.test_accuracy(verbose=True, score_type = 'coupling', adjust = 'csls')

        print('Results on test dictionary for fitting vectors: (via bary projection)')
        acc_dict['bary'] = BLI.test_accuracy(verbose=True, score_type = 'barycentric')
        print('Results on test dictionary for fitting vectors: (via bary projection + csls)')
        acc_dict['bary_csls'] = BLI.test_accuracy(verbose=True, score_type = 'barycentric', adjust = 'csls')

        print('Results on test dictionary for fitting vectors: (via orth projection)')
        acc_dict['proj'] = BLI.test_accuracy(verbose=True, score_type = 'projected')
        print('Results on test dictionary for fitting vectors: (via orth projection + csls)')
        acc_dict['proj_csls'] = BLI.test_accuracy(verbose=True, score_type = 'projected', adjust = 'csls')


    def testing(self, lang1, lang2):
        print("="*20+"begin testing mapping for "+lang1+" and "+lang2+"="*21)

        try:
            # 1. Solve Gromov Wasserstein problem
            coupling = self.get_lang_mapping(lang1, lang2)
            self.create_evaluation(lang1, lang2, coupling)
        except:
            print("failed to eval on "+lang1+" and "+lang2)
 
        
    def _get_shortest_path_from_lang1_to_lang2(self, lang1, lang2):
        lang1 = self.lang_dict[lang1]
        lang2 = self.lang_dict[lang2]
        lang1_to_root = []
        root_to_lang2 = []
        while lang1.lang != self.tree_root.lang:
            lang1_to_root.append((lang1.parent.lang, lang1.idx, True))
            lang1 = lang1.parent
        while lang2.lang != self.tree_root.lang:
            root_to_lang2.insert(0, (lang2.parent.lang, lang2.idx, False))
            lang2 = lang2.parent
        common_ancester = 0
        if len(lang1_to_root) == 0:
            return root_to_lang2
        if len(root_to_lang2) == 0:
            return lang1_to_root
        for i in range(0, len(root_to_lang2)):
            if (root_to_lang2[i][0] == lang1_to_root[len(lang1_to_root)-1-i][0] 
                    and root_to_lang2[i][1] == lang1_to_root[len(lang1_to_root)-1-i][1]):
                common_ancester = i+1
            else:
                break
        if common_ancester == 0:
            path = lang1_to_root+root_to_lang2
        else:
            path = lang1_to_root[:-common_ancester]+root_to_lang2[common_ancester:]
        return path

    def _load_word_embedding(self, lang):
        """
            Assumes file structure as in MUSE repo.
        """
        dict_fold = 'train' # which fold of the data will be used to produce results
        if self.args.task == 'conneau' or self.args.task == 'xling':
            data_dir = os.path.join(self.args.data_dir, 'MUSE')
            lang_path = os.path.join(data_dir, 'wiki.' + lang + '.vec')
        elif self.args.task == 'dinu':
            data_dir = os.path.join(self.args.data_dir, 'dinu')
            lang_path = os.path.join(data_dir, 'embeddings', lang + '.emb.txt')
        elif self.args.task == 'zhang':
            order = [lang,trg]
            if lang == 'en':
                order = order[::-1]
            data_dir = os.path.join(self.args.home_dir,'pkg/UBiLexAT/data/','-'.join(order))
            lang_path = os.path.join(data_dir, 'word2vec.' + lang)

        langfile = open(lang_path, encoding=self.args.encoding, errors='surrogateescape')
        words, xs = embeddings.read(langfile, self.args.maxs)
        langfile.close()
        # Build word to index map
        word2ind = {word: i for i, word in enumerate(words)}

        return xs, words, word2ind

    def read_embedding(self, lang):
        xs, _, _ =  self._load_word_embedding(lang)
        print("size of language set "+lang+" is:"+str(len(xs)))
        xs = self.normalize_embeddings(xs)
        xs = self._intra_language_distances(xs)
        return xs


    def normalize_embeddings(self, xs):
        if self.args.normalize_vecs:
            print("Normalizing embeddings with: {}".format(self.args.normalize_vecs))
        if self.args.normalize_vecs == 'whiten':
            print('Warning: whiten not yet implemented for OOV data')
            xs = self._center_embeddings(xs)
            xs = self._whiten_embeddings(xs)
        elif self.args.normalize_vecs == 'mean':
            xs = self._center_embeddings(xs)
        elif self.args.normalize_vecs == 'both':
            xs = self._center_embeddings(xs)
            xs = self._scale_embeddings(xs)
        elif self.args.normalize_vecs == 'whiten_zca':
            print('Warning: whiten zca not yet implemented for OOV data')
            xs = self._center_embeddings(xs)
            xs = self._whiten_embeddings_zca(xs)
        else:
            print('Warning: no normalization')
        return xs

    def _center_embeddings(self, xs):
        xs = xs - xs.mean(axis=0)
        return xs

    def _scale_embeddings(self, xs):
        xs = xs / np.linalg.norm(xs, axis=1)[:, None]
        return xs

    def _whiten_embeddings(self, xs):
        """
            PCA whitening. https://stats.stackexchange.com/questions/95806/how-to-whiten-the-data-using-principal-component-analysis
            Uses PCA of covariance matrix Sigma = XX', Sigma = ULU'.
            Whitening matrix given by:
                W = L^(-1/2)U'

        """

        n,d = xs.shape

        Cov_s = np.cov(xs.T)
        _, S_s, V_s = np.linalg.svd(Cov_s)
        W_s = (V_s.T/np.sqrt(S_s)).T
        assert np.allclose(W_s@Cov_s@W_s.T, np.eye(d)) # W*Sigma*W' = I_d

        xs = xs@W_s.T

        assert np.allclose(np.cov(xs.T), np.eye(d))  # Cov(hat(x)) = I_d
        return xs

    def _whiten_embeddings_zca(self, xs, lambd = 1e-8):
        """ ZCA whitening
            (C_xx+gamma I)^{-0.5}X
            (C_yy+gamma I)^{-0.5}Y
        """
        print('ZCA-Whitening')
        Cov_s = np.cov(xs.T)
        d = Cov_s.shape[0]

        W_s =  scipy.linalg.sqrtm(Cov_s + lambd*np.eye(d))
        xs = xs@W_s#.T
        return xs

    def _intra_language_distances(self, X):
        print('Computing intra-domain distance matrices...')

        if not self.gw_args.gpu:
            C1 = sp.spatial.distance.cdist(X, X, metric=self.gw_args.metric)
            if self.gw_args.normalize_dists == 'max':
                C1 /= C1.max()
            elif self.gw_args.normalize_dists == 'mean':
                C1 /= C1.mean()
            elif self.gw_args.normalize_dists == 'median':
                C1 /= np.median(C1)
        else:
            C1 = cdist(X, X, metric=self.gw_args.metric, returnAsGPU=True)
            if self.gw_args.normalize_dists == 'max':
                C1.divide(float(np.max(C1.asarray())))
            elif self.gw_args.normalize_dists == 'mean':
                C1.divide(float(np.mean(C1.asarray())))
            elif self.gw_args.normalize_dists == 'median':
                raise NotImplemented(
                    "Median normalization not implemented in GPU yet")

        stats_C1 = describe(C1.flatten())

        for (k, C, v) in [('C1', C1, stats_C1)]:
            print('Stats Distance Matrix {}. mean: {:8.2f}, median: {:8.2f},\
             min: {:8.2f}, max:{:8.2f}'.format(k, v.mean, np.median(C), v.minmax[0], v.minmax[1]))

        return C1

    def pprint_golddict(self, d, src, tgt):
        for i,vals in d.items():
            print(  '{:20s} <-> {:20s}'.format(src[i],','.join([tgt[i] for i in vals])))

    def zipf_init(self, lang, n):
        # See (Piantadosi, 2014)
        if lang == 'en':
            alpha, beta = 1.40, 1.88 #Other sources give: 1.13, 2.73
        elif lang == 'fi':
            alpha, beta = 1.17, 0.60
        elif lang == 'fr':
            alpha, beta = 1.71, 2.09
        elif lang == 'de':
            alpha, beta = 1.10, 0.40
        elif lang == 'es':
            alpha, beta = 1.84, 3.81
        else: # Deafult to EN
            alpha, beta = 1.40, 1.88
        p = np.array([1/((i+1)+beta)**(alpha) for i in range(n)])
        return p/p.sum()




class barycenter_mapping(multi_lang_embedding):
    def __init__(self, args, gw_args, outdir, mappingdir="tempdump/conneau"):
        self.args = args
        self.gw_args = gw_args
        self.outdir = outdir
        self.mappingdir = mappingdir
        try:
            name = args.tree.replace("-","_")
            self.tree_root, self.lang_dict, self.barycenter_in_seq = globals()[name]()
        except:
            raise NotImplemented(
                    "Please choose in implemented trees")

    def read_embedding(self, lang):
        xs, _, _ =  self._load_word_embedding(lang)
        print("size of language set "+lang+" is:" + str(len(xs)))
        xs = self.normalize_embeddings(xs)
        return xs

    def train(self):
        langs = self.lang_dict.keys()
        langs = list(filter(lambda x: not x.isdigit(), langs))

        for cnt in range(self.args.convergeiter):
            print("========== iteration "+str(cnt)+" ==========")
            # calculate space proj
            print("begin training space projection")
            self.train2()
            # calcluate the alignment
            print("recalculating the space alignment matrix")
            self.cal_alignment()

            for i in range(len(langs)):
                for j in range(len(langs)):
                    if i!=j and (not langs[i].isdigit()) and (not langs[j].isdigit()):
                        print("="*20+"begin testing mapping for "+langs[i]+" and "+langs[j]+"="*21)
                        self.testing(langs[i], langs[j])
        return langs


    def get_mapping(self, coupling, xs, xt):
        best_match_src = coupling.argmax(1) # Best match for each source word
        best_match_trg = coupling.argmax(0) # Best match for each source word

        ns = len(coupling)
        paired = []
        for i in range(ns):
            m = best_match_src[i]
            if best_match_trg[m] == i:
                paired.append((i,m))

        idx_src = [ws for ws,_ in paired]
        idx_trg = [wt for _,wt in paired]

        xs_nn = xs[idx_src]
        xt_nn = xt[idx_trg]
        P = orth_procrustes(xs_nn, xt_nn)
        return P


    def cal_alignment(self):
        # get mapping
        lang2 = self.args.lang_space
        for lang1 in self.lang_dict.keys():
            if lang1 != lang2 and (not lang1.isdigit()):
                # get coupling matrix for lang1 to lang2
                coupling = self.get_lang_mapping(lang1, lang2, self.gw_args.metric, self.gw_args.entreg)                

                node1 = self.lang_dict[lang1]
                node1.load_barycenter_mtrx(self.outdir, self)
                xs = node1.barycenter_mtrx
                node2 = self.lang_dict[lang2]
                node2.load_barycenter_mtrx(self.outdir, self)
                xt = node2.barycenter_mtrx

                # use Procrustes matching to get alignment matrix
                P = self.get_mapping(coupling, xs, xt)

                # save projection matrix
                dirname = lang1 + "_" + lang2 + "_5000"
                project_mtrx_path = os.path.join(self.outdir, dirname)
                if not os.path.exists(project_mtrx_path):
                    os.makedirs(project_mtrx_path)
                project_mtrx_path = os.path.join(project_mtrx_path, "mapping.npy")
                np.save(project_mtrx_path, P)

    def train2(self):
        for barycenter in self.barycenter_in_seq:
            self.train_barycenter(barycenter)

    def train_barycenter(self, barycenter, lang=None):
        print('training barycenter ' + barycenter.lang)
        if lang is None:
            lang = self.args.lang_space
        if not barycenter.lang.isdigit():
            return

        Cs = []
        ps = []
        ls = []
        dim = 0
        # step 1: project all into one space
        for child in barycenter.child:
            self.project_into_lang_space(child, lang)
            Cs.append(child.projected_matrix)
            ps.append(child.freq)
            ls.append(child.lang)
            dim += len(child.projected_matrix)
         
        # step 2: train barycenter

        save_path = os.path.join(self.outdir, "barycenter_"+str(barycenter.lang)+".npy")
        print(save_path)

        #if os.path.exists(save_path):
        #    d = load_dict(save_path)
        #    if self.args.option == "barycenter":
        #        if d is not None and 'project_mtrx' in d.keys():
        #            barycenter.projected_matrix, barycenter.freq = d['project_mtrx'], d['freq']
        #        barycenter.projected_matrix, barycenter.freq = cal_barycenter_for(
        #                ls, #ls
        #                self.create_evaluation, #test_func
        #                Cs, #Cs
        #                ps, #ps
        #                ot.unif(len(ls)), # lambdas
        #                barycenter.projected_matrix, #X_init
        #                tol=self.gw_args.tol, metric=self.gw_args.metric, reg=self.gw_args.entreg, bregmanmaxiter=self.gw_args.bregmanmaxiter, maxiter=self.args.maxiter)
        #    else:
        #        if d is not None and 'project_mtrx' in d.keys():
        #            barycenter.projected_matrix, barycenter.freq = d['project_mtrx'], d['freq']
        #            xinit = d['project_mtrx']
        #        barycenter.projected_matrix, barycenter.freq = unweighted_barycenter(xinit, Cs, ps, tol=self.gw_args.tol, metric=self.gw_args.metric, reg=self.gw_args.entreg, maxiter=self.args.maxiter)

        #else:
        if self.args.dim == "sum":
            dim += 0
        elif self.args.dim == "2times":
            dim = 2*len(Cs[0])
        else:
            dim = len(Cs[0])
 
        if self.args.initlang != 'random':
            xinit = Cs[ls.index(self.args.initlang)]
        else:
            xinit = np.random.rand(dim, len(Cs[0][0]))


        if self.args.option == "barycenter":
            barycenter.projected_matrix, barycenter.freq = cal_barycenter_for(
                    ls, self.create_evaluation, Cs, ps,
                    ot.unif(len(barycenter.child)),
                    xinit,
                    tol=self.gw_args.tol, metric=self.gw_args.metric, reg=self.gw_args.entreg, bregmanmaxiter=self.gw_args.bregmanmaxiter, maxiter=self.args.maxiter)

        elif self.args.option == "unw":
            barycenter.projected_matrix, barycenter.freq = unweighted_barycenter(xinit, Cs, ps, tol=self.gw_args.tol, metric=self.gw_args.metric, reg=self.gw_args.entreg, maxiter=self.args.maxiter)

        # step 3: save it
        barycenter_info = {"project_mtrx": barycenter.projected_matrix, "freq": barycenter.freq}
        save_dict(barycenter_info, save_path)

        Cs = None
        barycenter.projected_matrix = None
        for c in barycenter.child:
            c.projected_matrix = None

        return self


    def project_into_lang_space(self, node, lang):
        if not node.lang.isdigit():
            node.load_barycenter_mtrx(self.outdir, self)
            project_mtrx = self._get_proj_matrix(node, lang)

            if project_mtrx is None:
                node.projected_matrix = node.barycenter_mtrx
            else:
                node.projected_matrix = np.matmul(node.barycenter_mtrx, project_mtrx)
                
            node.barycenter_mtrx = None
            node.mapping_to_child = None

        else:
            save_path = os.path.join(self.outdir, "barycenter_"+str(node.lang)+".npy")
            if os.path.exists(save_path):
                loaddict = load_dict(save_path)
                node.projected_matrix = loaddict["project_mtrx"]
                node.freq = loaddict["freq"]
            else:
                return None
        return node


    def _get_proj_matrix(self, node, lang):
        if node.lang == lang:
            return None

        dirname = node.lang + "_" + lang + "_5000"
        project_mtrx_path = os.path.join(self.outdir, dirname)
        project_mtrx_path = os.path.join(project_mtrx_path, "mapping.npy")
        if os.path.exists(project_mtrx_path):
            print("~loading projection matrix @" + str(project_mtrx_path))
            return np.transpose(np.load(project_mtrx_path))

        if self.args.task == 'dinu':
            project_mtrx_path = os.path.join(self.mappingdir, dirname)
            project_mtrx_path = node.lang + "_" + lang + ".npy"
            if os.path.exists(project_mtrx_path):
                project_mtrx = np.load(project_mtrx_path)
            else:
                project_mtrx_path = lang + "_" + node.lang + ".npy"
                if not os.path.exists(project_mtrx_path):
                    return None
                else:
                    project_mtrx = np.load(project_mtrx_path)
                    project_mtrx = np.transpose(project_mtrx)

        elif self.args.task == 'conneau' or self.args.task == 'xling':
            dirname = node.lang + "_" + lang + "_5000"
            project_mtrx_path = os.path.join(self.mappingdir, dirname)
            project_mtrx_path = os.path.join(project_mtrx_path, "gromov_both_cosine_uniform_ereg_0.0001_tol_1e-08")
            project_mtrx_path = os.path.join(project_mtrx_path, "mapping.npy")

            if os.path.exists(project_mtrx_path):
                project_mtrx = np.load(project_mtrx_path)
            else:
                dirname = lang + "_" + node.lang + "_5000"
                project_mtrx_path = os.path.join(self.mappingdir, dirname)
                project_mtrx_path = os.path.join(project_mtrx_path, "gromov_both_cosine_uniform_ereg_0.0001_tol_1e-08")
                project_mtrx_path = os.path.join(project_mtrx_path, "mapping.npy")
                if os.path.exists(project_mtrx_path):
                    project_mtrx = np.transpose(np.load(project_mtrx_path))
                else:
                    return None
        print("loading from init")

        return project_mtrx


    def get_lang_mapping(self, lang1, lang2, metric, entreg):
        path = self._get_shortest_path_from_lang1_to_lang2(lang1, lang2)
        print(path)

        mapping = None
        for i in range(len(path)):
            a = self.lang_dict[path[i][0]]
            a = self.project_into_lang_space(a, self.args.lang_space)
            b = self.lang_dict[path[i][0]].child[path[i][1]]
            b = self.project_into_lang_space(b, self.args.lang_space)

            if path[i][2]:
                plan = ot.emd(b.freq, a.freq, build_MXY(b.projected_matrix, a.projected_matrix))
            else:
                plan = ot.emd(a.freq, b.freq, build_MXY(a.projected_matrix, b.projected_matrix))

            if mapping is None:
                mapping = plan
            else:
                mapping = np.matmul(mapping, plan)

            a.projected_matrix = None
            b.projected_matrix = None

        return mapping


    def testing(self, lang1, lang2):
        print("="*20+"begin testing mapping for "+lang1+" and "+lang2+"="*21)
        try:
            # 1. Solve Gromov Wasserstein problem
            coupling = self.get_lang_mapping(lang1, lang2, self.gw_args.metric, self.gw_args.entreg)
            self.create_evaluation(lang1, lang2, coupling)
        except:
            print("failed to eval on "+lang1+" and "+lang2)




