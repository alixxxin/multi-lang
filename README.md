
==============================
# BAlign

Code for paper ["Unsupervised Multilingual Alignment using Wasserstein Barycenter"](https://www.ijcai.org/Proceedings/2020/512). 

Please cite this work as:

```
@inproceedings{ijcai2020-512,
  title     = {Unsupervised Multilingual Alignment using Wasserstein Barycenter},
  author    = {Lian, Xin and Jain, Kshitij and Truszkowski, Jakub and Poupart, Pascal and Yu, Yaoliang},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {3702--3708},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/512},
  url       = {https://doi.org/10.24963/ijcai.2020/512},
}
```


Disclaimer: This codebase borrows some embbedding and evaluation tools from David Alvarez-Melis's [otalign](https://github.com/dmelis/otalign), Mikel Artetxe's [vecmap](https://github.com/artetxem/vecmap) repo, and Python Optimal Transport [POT](https://github.com/rflamary/POT) from Remi Flamary and colleagues.

## Dependencies

#### Major:
* python (>3.0)
* numpy (>1.15)
* scipy
* matplotlib
* pylab
* [POT](https://github.com/rflamary/POT) (>0.5)
* (OPTIONAL) [cupy](https://cupy.chainer.org) (for GPU computation)

#### Minor
* tqdm
* matplotlib

## Installation

It's highly recommended that the following steps be done **inside a virtual environment** (e.g., via `virtualenv` or `anaconda`).

Install requirements
```
pip3 install -r requirements.txt
```

Copy data to local dirs (Optional, can also be specified via arguments)

```
cp -r /path/to/MUSE/dir/data/* ./data/raw/MUSE/
cp -r /path/to/dinu/dir/data/* ./data/raw/dinu/

```

## How to use
We implemented three different methods to align all languages simultaneously: Barycenter approach, GW-Barycenter approach and unweighted approach (equivalent to nearest neighbour).
We also added the functionality to construct a hierarchical tree and train hierarchical barycenters to imply language mappings. With different trees, the training process will train all barycenter nodes.
We can choose between those methods and language tree struture by setting parameters `--option` and `--tree`.
There are three dataset we can evaluate our results on: 'Conneau', 'XLING', and 'dinu'. 

Example command to use the package: the following command run against the XLING dataset with barycenter method initialize with 2 times the size of each vocabulary

```
python3 -u scripts/main_gw_mli.py --task xling --entreg 10 --maxiter 20 --maxs 5000 --tree test-tree --lang_space fr --results_path out8/xling-test-unif-dist20000 --option barycenter --initlang random --dim 2times
```

```
## Command line options:
| Option name | parameter |
| data_dir | location word embedding and evaluation datasets are put in |
| distribs | distribution type for word embedding weights  |
| encoding | encoding for words  |
| entropic | use Wasserstein distance if the flag is true, use linear program to solve OT exactly otherwise  |
| entreg | the entropy regularizer we use for Wasserstein problem |
| tol | tolerance threshold |
| maxs | number of words in each vocabulary we use for training |
| metric | similarity function we use to measure similiarity between words |
| results_path | location to put results in |
| task | dataset to evaluate against |
| lang_space | language space to map all languages into |
| normalize_dists | method to normalize all word embeddings |
```
