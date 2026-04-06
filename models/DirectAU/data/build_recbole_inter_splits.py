#!/usr/bin/env python
"""
Build fixed RecBole atomic .inter splits from LightGCN-style train.txt/test.txt.

Input format (train.txt/test.txt):
    user_id item_id_1 item_id_2 ... item_id_n

Output files per dataset directory:
    <dataset>.train.inter
    <dataset>.valid.inter
    <dataset>.test.inter

Each .inter file schema:
    user_id:token\titem_id:token\ttimestamp:float

Usage:
    python build_recbole_inter_splits.py --dataset gowalla
    python build_recbole_inter_splits.py --dataset all --valid_ratio 0.1 --seed 2020
"""

import argparse
import os
import random
from collections import defaultdict


DEFAULT_DATASETS = ['movielens', 'gowalla', 'yelp2018', 'amazon-book', 'collected']


def read_lightgcn_file(path):
    """Read LightGCN-style user-item lists."""
    user_items = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            user_items[uid].extend(items)
    return dict(user_items)


def split_train_valid_fixed(train_user_items, valid_ratio=0.1, seed=2020):
    """Create deterministic train/valid split from train interactions per user."""
    train_split = {}
    valid_split = {}

    for uid in sorted(train_user_items.keys()):
        items = list(train_user_items[uid])
        if len(items) <= 1:
            train_split[uid] = items
            valid_split[uid] = []
            continue

        rng = random.Random(seed + uid)
        rng.shuffle(items)

        n_valid = max(1, int(len(items) * valid_ratio))
        n_valid = min(n_valid, len(items) - 1)

        valid_items = items[:n_valid]
        train_items = items[n_valid:]

        train_split[uid] = train_items
        valid_split[uid] = valid_items

    return train_split, valid_split


def to_interactions(user_items, start_ts=1):
    """Convert user->items map to list of (uid, iid, timestamp)."""
    interactions = []
    ts = start_ts
    for uid in sorted(user_items.keys()):
        for iid in user_items[uid]:
            interactions.append((uid, iid, float(ts)))
            ts += 1
    return interactions, ts


def write_inter_file(path, interactions):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        for uid, iid, ts in interactions:
            f.write(f'{uid}\t{iid}\t{ts:.1f}\n')


def build_for_dataset(dataset_dir, dataset_name, valid_ratio, seed):
    train_txt = os.path.join(dataset_dir, 'train.txt')
    test_txt = os.path.join(dataset_dir, 'test.txt')
    if not (os.path.exists(train_txt) and os.path.exists(test_txt)):
        raise FileNotFoundError(f'Missing train/test txt for {dataset_name} in {dataset_dir}')

    train_user_items = read_lightgcn_file(train_txt)
    test_user_items = read_lightgcn_file(test_txt)

    train_split, valid_split = split_train_valid_fixed(
        train_user_items, valid_ratio=valid_ratio, seed=seed
    )

    train_inter, ts = to_interactions(train_split, start_ts=1)
    valid_inter, ts = to_interactions(valid_split, start_ts=ts)
    test_inter, _ = to_interactions(test_user_items, start_ts=ts)

    # Safety fallback: valid split must not be empty for RecBole 3-way unpacking.
    if not valid_inter:
        moved = False
        for uid in sorted(train_split.keys()):
            if len(train_split[uid]) > 1:
                iid = train_split[uid].pop()
                valid_split[uid].append(iid)
                moved = True
                break
        if moved:
            train_inter, ts = to_interactions(train_split, start_ts=1)
            valid_inter, ts = to_interactions(valid_split, start_ts=ts)
            test_inter, _ = to_interactions(test_user_items, start_ts=ts)

    out_train = os.path.join(dataset_dir, f'{dataset_name}.train.inter')
    out_valid = os.path.join(dataset_dir, f'{dataset_name}.valid.inter')
    out_test = os.path.join(dataset_dir, f'{dataset_name}.test.inter')

    write_inter_file(out_train, train_inter)
    write_inter_file(out_valid, valid_inter)
    write_inter_file(out_test, test_inter)

    print(f'[{dataset_name}] train={len(train_inter)} valid={len(valid_inter)} test={len(test_inter)}')
    print(f'  -> {out_train}')
    print(f'  -> {out_valid}')
    print(f'  -> {out_test}')


def main():
    parser = argparse.ArgumentParser(description='Build RecBole .inter fixed splits from train/test txt')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset name or "all"')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of train interactions per user moved to valid')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Seed for deterministic per-user split')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == 'all':
        datasets = DEFAULT_DATASETS
    else:
        datasets = [args.dataset]

    for ds in datasets:
        ds_dir = os.path.join(script_dir, ds)
        try:
            build_for_dataset(ds_dir, ds, valid_ratio=args.valid_ratio, seed=args.seed)
        except Exception as exc:
            print(f'[SKIP] {ds}: {exc}')


if __name__ == '__main__':
    main()
