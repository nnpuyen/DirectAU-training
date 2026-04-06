# Data Collection & Preprocessing Documentation

> This document describes how all 4 benchmark datasets were obtained from their **official sources** and preprocessed into the unified format used by our experiments (LightGCN, DirectAU, BERT4Rec).

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Sources](#dataset-sources)
3. [Download Instructions](#download-instructions)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Output Format](#output-format)
6. [Final Dataset Statistics](#final-dataset-statistics)
7. [How to Reproduce](#how-to-reproduce)

---

## Overview

We use **4 benchmark datasets** from official academic/public sources, covering different domains:

| Dataset | Domain | Official Source |
|---------|--------|-----------------|
| MovieLens 1M | Movie ratings | GroupLens Research |
| Gowalla | Location check-ins | Stanford SNAP |
| Yelp2018 | Business reviews | Yelp Open Dataset |
| Amazon-Book | Book purchases | Amazon Review Data (UCSD) |

All datasets are converted to **implicit feedback** (binary 0/1: interacted or not) and stored in a unified `train.txt` / `test.txt` format.

---

## Dataset Sources

### 1. MovieLens 1M

- **Official page**: https://grouplens.org/datasets/movielens/1m/
- **Direct download**: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- **Citation**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
- **Raw format**: `ratings.dat` — each line is `UserID::MovieID::Rating::Timestamp`
- **Scale**: 1,000,209 ratings from 6,040 users on 3,706 movies
- **Size**: ~6 MB (zip)

### 2. Gowalla

- **Official page**: https://snap.stanford.edu/data/loc-gowalla.html
- **Direct downloads**:
  - `loc-gowalla_totalCheckins.txt.gz` (check-in data)
  - `loc-gowalla_edges.txt.gz` (social network)
- **Citation**: E. Cho, S. A. Myers, J. Leskovec. Friendship and Mobility: User Movement in Location-Based Social Networks. KDD 2011.
- **Raw format**: Each line is `user_id \t check-in_time \t latitude \t longitude \t location_id`
- **Scale**: 6,442,890 check-ins from 196,591 users
- **Size**: ~100 MB (gzipped)

### 3. Yelp2018

- **Official page**: https://business.yelp.com/data/resources/open-dataset/
- **Direct download**: https://business.yelp.com/external-assets/files/Yelp-JSON.zip
- **Citation**: Yelp Dataset Challenge (used in many publications)
- **Raw format**: JSON lines in `yelp_academic_dataset_review.json` — each line is a JSON object with `user_id`, `business_id`, `stars`, etc.
- **Scale**: ~6,990,280 reviews from ~1,987,929 users on 150,346 businesses
- **Size**: ~4.35 GB (zip) → ~5 GB review JSON

### 4. Amazon-Book

- **Official page**: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- **Alternative mirror**: https://nijianmo.github.io/amazon/index.html
- **Download**: Ratings-only CSV for the "Books" category
- **Citation**: Jianmo Ni, Jiacheng Li, Julian McAuley. Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. EMNLP 2019.
- **Raw format**: CSV with columns `item_id, user_id, rating, timestamp`
- **Scale**: ~51,311,621 ratings from ~15,362,619 users on ~2,930,451 books
- **Size**: ~2 GB (CSV)

---

## Download Instructions

All downloads were performed using PowerShell `Invoke-WebRequest` from the official URLs listed above.

### MovieLens 1M

```powershell
cd data\movielens
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-1m.zip" -OutFile "ml-1m.zip"
Expand-Archive -Path "ml-1m.zip" -DestinationPath "." -Force
```

After extraction, the `ml-1m/` folder contains:
- `ratings.dat` — the main ratings file (24 MB)
- `movies.dat` — movie metadata
- `users.dat` — user demographics
- `README` — dataset documentation

### Gowalla

```powershell
cd data\gowalla
Invoke-WebRequest -Uri "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz" -OutFile "loc-gowalla_totalCheckins.txt.gz"
Invoke-WebRequest -Uri "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz" -OutFile "loc-gowalla_edges.txt.gz"
```

Files are gzip-compressed; the preprocessing script reads `.gz` files directly.

### Yelp2018

```powershell
cd data\yelp2018
Invoke-WebRequest -Uri "https://business.yelp.com/external-assets/files/Yelp-JSON.zip" -OutFile "Yelp-JSON.zip"
Expand-Archive -Path "Yelp-JSON.zip" -DestinationPath "." -Force
# The zip contains a tar archive inside a "Yelp JSON" folder
cd "Yelp JSON"
tar -xf yelp_dataset.tar
```

After extraction, the key file is `yelp_academic_dataset_review.json` (~5 GB).

### Amazon-Book

```powershell
cd data\amazon-book
Invoke-WebRequest -Uri "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books.csv" -OutFile "Books_ratings.csv"
```

This downloads the **ratings-only** CSV file (~2 GB) for the Books category.

---

## Preprocessing Pipeline

All raw datasets go through the same preprocessing pipeline implemented in [`preprocess_all.py`](preprocess_all.py):

### Step 1: Load Raw Data → Implicit Feedback

Each dataset has a specific loader that reads the raw format and converts it to implicit feedback:

| Dataset | Raw Signal | Conversion |
|---------|-----------|------------|
| MovieLens 1M | Explicit ratings (1–5 stars) | Any rating → interaction = 1 |
| Gowalla | Check-in events | Any check-in at a location → interaction = 1 (deduplicated per user-location pair) |
| Yelp2018 | Review with star rating | Any review → interaction = 1 |
| Amazon-Book | Rating (1–5) | Any rating → interaction = 1 |

**Key decisions:**
- All explicit ratings are binarized: if a user interacted with an item at all, it counts as a positive signal (= 1).
- For Gowalla, multiple check-ins at the same location by the same user are deduplicated (using a set).
- String-based user/item IDs (Yelp, Amazon) are mapped to integer IDs during loading.

### Step 2: K-Core Filtering

We apply **10-core filtering** iteratively: only keep users and items that each have at least 10 interactions. This is repeated until convergence (no more users/items are removed).

**Why k-core filtering?**
- Removes very cold users/items that don't have enough data for meaningful recommendations.
- Produces a denser interaction matrix, which is standard practice in recommendation research.
- k=10 is a common threshold used in LightGCN, DirectAU, and other papers.

**Effect of filtering:**

| Dataset | Before (users) | After (users) | Before (items) | After (items) | Before (interactions) | After (interactions) |
|---------|---------------:|-------------:|---------------:|-------------:|---------------------:|--------------------:|
| MovieLens 1M | 6,040 | 6,040 | 3,706 | 3,260 | 1,000,209 | 998,539 |
| Gowalla | 107,092 | 29,858 | — | 40,988 | 3,981,334 | 1,027,464 |
| Yelp2018 | 1,987,929 | 93,537 | 150,346 | 53,347 | 6,745,760 | 2,533,759 |
| Amazon-Book | 15,362,619 | 603,617 | 2,930,451 | 298,833 | 51,062,224 | 16,617,185 |

### Step 3: ID Remapping

After filtering, user and item IDs are remapped to **contiguous integers starting from 0**. This is required because:
- LightGCN (and most models) use IDs as indices into embedding matrices.
- Sparse IDs would waste memory and cause index errors.

### Step 4: Train/Test Split

Each user's interactions are split into **80% train / 20% test**:
- Items are shuffled randomly (with fixed seed = 2020 for reproducibility).
- At least 1 item per user is always placed in the test set.
- This is a **random split** strategy (not leave-one-out or temporal).

### Step 5: Save to LightGCN Format

The output files follow the format used by the LightGCN codebase:
- Each line: `user_id item_id_1 item_id_2 ... item_id_n`
- Separate files for `train.txt` and `test.txt`

---

## Output Format

Each dataset directory contains:

```
data/<dataset>/
├── train.txt      # Training interactions (LightGCN format)
├── test.txt       # Test interactions (LightGCN format)
├── stats.txt      # Summary statistics
└── <raw files>    # Original downloaded files
```

**Format of `train.txt` / `test.txt`:**
```
0 45 123 678 901 ...
1 23 456 789 ...
2 12 34 56 78 90 ...
...
```
Each line starts with the user ID, followed by the item IDs that user interacted with.

---

## Final Dataset Statistics

| Dataset | Users | Items | Train | Test | Total | Sparsity |
|---------|------:|------:|------:|-----:|------:|---------:|
| **MovieLens 1M** | 6,040 | 3,260 | 801,218 | 197,321 | 998,539 | 94.93% |
| **Gowalla** | 29,858 | 40,988 | 833,031 | 194,433 | 1,027,464 | 99.92% |
| **Yelp2018** | 93,537 | 53,347 | 2,059,466 | 474,293 | 2,533,759 | 99.95% |
| **Amazon-Book** | 603,617 | 298,833 | 13,502,081 | 3,115,104 | 16,617,185 | 99.99% |

### Observations:
- **MovieLens 1M** is the smallest and densest — good for quick experimentation and debugging.
- **Gowalla** is medium-scale with high sparsity — a standard graph-based recommendation benchmark.
- **Yelp2018** is larger with ~2.5M interactions — tests scalability while remaining manageable.
- **Amazon-Book** is the largest (~16.6M interactions) — the most challenging dataset for training time and memory.

---

## How to Reproduce

### Prerequisites

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install numpy
```

### Download all datasets

Follow the download instructions above for each dataset, or use the commands listed.

### Run preprocessing

```bash
cd data

# Preprocess individual datasets
python preprocess_all.py --dataset movielens
python preprocess_all.py --dataset gowalla
python preprocess_all.py --dataset yelp2018
python preprocess_all.py --dataset amazon-book

# Or preprocess all at once
python preprocess_all.py --dataset all

# Custom parameters
python preprocess_all.py --dataset movielens --k-core 5 --test-ratio 0.1
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | (required) | `movielens`, `gowalla`, `yelp2018`, `amazon-book`, or `all` |
| `--k-core` | 10 | Minimum interactions per user/item |
| `--test-ratio` | 0.2 | Fraction of each user's items held out for testing |

### Random Seed

All random operations use seed **2020** for full reproducibility. Running the same command twice will produce identical `train.txt` and `test.txt` files.

---

## Tools & Environment Used

- **OS**: Windows
- **Python**: 3.12.8
- **Dependencies**: `numpy` (for random shuffling and array operations)
- **Download tool**: PowerShell `Invoke-WebRequest`
- **Archive tools**: PowerShell `Expand-Archive`, `tar`
