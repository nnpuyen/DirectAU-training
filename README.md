# Recommendation System Experiments

> CS14114 | Big Data Applications — Course Project: Recommendation System with Implicit Feedback

## Project Structure

```
Recommendation System Experiment/
├── README.md
├── data/                          # Datasets (5 total)
│   ├── gowalla/                   # Benchmark dataset 1 (SNAP Stanford)
│   ├── yelp2018/                  # Benchmark dataset 2 (Yelp Open Dataset)
│   ├── amazon-book/               # Benchmark dataset 3 (Amazon Review 2018)
│   ├── movielens/                 # Benchmark dataset 4 (MovieLens 1M)
│   ├── collected/                 # Self-collected dataset (Phase 2)
│   └── preprocess_all.py          # Preprocessing script for all datasets
├── models/                        # Model implementations (3 SOTA)
│   ├── LightGCN/                  # LightGCN (cloned from gusye1234/LightGCN-PyTorch)
│   ├── DirectAU/                  # DirectAU (TODO)
│   └── BERT4Rec/                  # BERT4Rec (TODO)
└── results/                       # Experiment results
    ├── LightGCN/
    ├── DirectAU/
    └── BERT4Rec/
```

## Models

| Model | Paper | Year | Family | Repository |
|-------|-------|------|--------|------------|
| **LightGCN** | Simplifying and Powering Graph Convolution Network for Recommendation | SIGIR 2020 | Graph-based | [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch) |
| **DirectAU** | Towards Representation Alignment and Uniformity in Collaborative Filtering | KDD 2022 | Representation Learning | [THUwangcy/DirectAU](https://github.com/THUwangcy/DirectAU) |
| **BERT4Rec** | Sequential Recommendation with BERT | CIKM 2019 | Sequential / Transformer | [FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec) |

## Datasets (from Official Sources)

| # | Dataset | Domain | Official Source | Download |
|---|---------|--------|-----------------|----------|
| 1 | **Gowalla** | Location check-ins | [SNAP Stanford](https://snap.stanford.edu/data/loc-gowalla.html) | Direct download |
| 2 | **Yelp2018** | Business reviews | [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) | Direct download |
| 3 | **Amazon-Book** | Book purchases | [Amazon Review 2018 (UCSD)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) | Ratings CSV |
| 4 | **MovieLens 1M** | Movie ratings | [GroupLens](https://grouplens.org/datasets/movielens/1m/) | Direct download |
| 5 | *Collected* | TBD | Self-collected (Phase 2) | — |

## Data Preprocessing

All raw datasets are converted to LightGCN format (`train.txt` / `test.txt`) using:

```bash
cd data
python preprocess_all.py --dataset movielens    # or gowalla, yelp2018, amazon-book, all
```

Each line in the output: `user_id item_id_1 item_id_2 ... item_id_n`

## Experiment Plan

- **3 models × 5 datasets = 15 experiments**
- Each experiment: run 3–5 times, report average
- Metrics: Precision@K, Recall@K, NDCG@K (K=20)

## Quick Start — LightGCN

```bash
cd models/LightGCN
pip install -r requirements.txt
cd code
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="movielens" --topks="[20]" --recdim=64
```

