"""
Preprocessing script for all 4 benchmark datasets.
Converts raw data from official sources into LightGCN format:
  train.txt / test.txt
  Each line: user_id item_id_1 item_id_2 ... item_id_n

Official Sources:
  - MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
  - Gowalla:      https://snap.stanford.edu/data/loc-gowalla.html
  - Yelp2018:     https://business.yelp.com/data/resources/open-dataset/
  - Amazon-Book:  https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Usage:
  python preprocess_all.py --dataset movielens
  python preprocess_all.py --dataset gowalla
  python preprocess_all.py --dataset yelp2018
  python preprocess_all.py --dataset amazon-book
  python preprocess_all.py --dataset all
"""

import os
import argparse
import gzip
import json
import random
import numpy as np
from collections import defaultdict

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)

# Minimum interactions per user/item to keep (k-core filtering)
K_CORE = 10
TEST_RATIO = 0.2  # fraction of interactions per user held out for test


def k_core_filter(user_items, k=10):
    """Iteratively remove users and items with fewer than k interactions."""
    changed = True
    while changed:
        changed = False
        # Filter items
        item_count = defaultdict(int)
        for items in user_items.values():
            for item in items:
                item_count[item] += 1
        valid_items = {item for item, cnt in item_count.items() if cnt >= k}

        new_user_items = {}
        for user, items in user_items.items():
            filtered = [i for i in items if i in valid_items]
            if len(filtered) >= k:
                new_user_items[user] = filtered
            else:
                changed = True

        if len(new_user_items) != len(user_items):
            changed = True
        # Check if items changed
        new_item_count = defaultdict(int)
        for items in new_user_items.values():
            for item in items:
                new_item_count[item] += 1
        if len(new_item_count) != len(valid_items):
            changed = True

        user_items = new_user_items
    return user_items


def remap_ids(user_items):
    """Remap user and item IDs to contiguous integers starting from 0."""
    all_users = sorted(user_items.keys())
    all_items = sorted({item for items in user_items.values() for item in items})

    user_map = {u: idx for idx, u in enumerate(all_users)}
    item_map = {i: idx for idx, i in enumerate(all_items)}

    remapped = {}
    for user, items in user_items.items():
        remapped[user_map[user]] = [item_map[i] for i in items]

    return remapped, user_map, item_map


def split_train_test(user_items, test_ratio=0.2):
    """Split each user's items into train and test sets."""
    train = {}
    test = {}
    for user, items in user_items.items():
        items = list(items)
        np.random.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test[user] = items[:n_test]
        train[user] = items[n_test:]
    return train, test


def write_lightgcn_format(user_items, filepath):
    """Write data in LightGCN format: each line is 'user_id item1 item2 ...'"""
    with open(filepath, 'w') as f:
        for user in sorted(user_items.keys()):
            items = user_items[user]
            line = str(user) + ' ' + ' '.join(str(i) for i in items)
            f.write(line + '\n')


def save_dataset(user_items, output_dir, dataset_name, k_core=10, test_ratio=0.2):
    """Full pipeline: k-core filter -> remap -> split -> save."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    total_interactions = sum(len(v) for v in user_items.values())
    print(f"Raw: {len(user_items)} users, {total_interactions} interactions")

    # K-core filtering
    user_items = k_core_filter(user_items, k=k_core)
    total_interactions = sum(len(v) for v in user_items.values())
    n_items = len({item for items in user_items.values() for item in items})
    print(f"After {k_core}-core: {len(user_items)} users, {n_items} items, {total_interactions} interactions")

    # Remap
    user_items, user_map, item_map = remap_ids(user_items)
    print(f"Remapped to 0-indexed: {len(user_map)} users, {len(item_map)} items")

    # Split
    train, test = split_train_test(user_items, test_ratio=test_ratio)
    train_interactions = sum(len(v) for v in train.values())
    test_interactions = sum(len(v) for v in test.values())
    print(f"Train: {train_interactions} interactions")
    print(f"Test:  {test_interactions} interactions")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    write_lightgcn_format(train, os.path.join(output_dir, 'train.txt'))
    write_lightgcn_format(test, os.path.join(output_dir, 'test.txt'))
    print(f"Saved to {output_dir}/train.txt and test.txt")

    # Save stats
    with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Users: {len(user_map)}\n")
        f.write(f"Items: {len(item_map)}\n")
        f.write(f"Train interactions: {train_interactions}\n")
        f.write(f"Test interactions: {test_interactions}\n")
        f.write(f"Total interactions: {train_interactions + test_interactions}\n")
        f.write(f"Sparsity: {1 - (train_interactions + test_interactions) / (len(user_map) * len(item_map)):.6f}\n")
    print(f"Stats saved to {output_dir}/stats.txt")


# ============================================================
# Dataset-specific loaders
# ============================================================

def load_movielens(data_dir):
    """Load MovieLens 1M from ratings.dat (UserID::MovieID::Rating::Timestamp)."""
    ratings_file = os.path.join(data_dir, 'ml-1m', 'ratings.dat')
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"{ratings_file} not found.\n"
            "Download from: https://files.grouplens.org/datasets/movielens/ml-1m.zip\n"
            "Extract into data/movielens/"
        )

    user_items = defaultdict(list)
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            user_id = int(parts[0])
            item_id = int(parts[1])
            # Implicit feedback: treat all ratings as positive interactions
            user_items[user_id].append(item_id)

    print(f"Loaded MovieLens 1M: {len(user_items)} users")
    return dict(user_items)


def load_gowalla(data_dir):
    """Load Gowalla check-ins from SNAP (user \\t time \\t lat \\t lon \\t location_id)."""
    checkin_file = os.path.join(data_dir, 'loc-gowalla_totalCheckins.txt.gz')
    checkin_txt = os.path.join(data_dir, 'loc-gowalla_totalCheckins.txt')

    if os.path.exists(checkin_file):
        open_fn = lambda: gzip.open(checkin_file, 'rt', encoding='utf-8')
    elif os.path.exists(checkin_txt):
        open_fn = lambda: open(checkin_txt, 'r', encoding='utf-8')
    else:
        raise FileNotFoundError(
            f"Gowalla data not found in {data_dir}.\n"
            "Download from: https://snap.stanford.edu/data/loc-gowalla.html\n"
            "Files needed: loc-gowalla_totalCheckins.txt.gz"
        )

    user_items = defaultdict(set)  # use set to deduplicate check-ins
    with open_fn() as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            user_id = int(parts[0])
            location_id = int(parts[4])
            user_items[user_id].add(location_id)

    # Convert sets to lists
    user_items = {u: list(items) for u, items in user_items.items()}
    print(f"Loaded Gowalla: {len(user_items)} users")
    return user_items


def load_yelp2018(data_dir):
    """Load Yelp dataset from the official JSON file."""
    # The Yelp download extracts to a folder with yelp_academic_dataset_review.json
    possible_paths = [
        os.path.join(data_dir, 'yelp_academic_dataset_review.json'),
        os.path.join(data_dir, 'Yelp-JSON', 'yelp_academic_dataset_review.json'),
        os.path.join(data_dir, 'yelp_dataset', 'yelp_academic_dataset_review.json'),
    ]
    # Also look for any .tar extraction
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname == 'yelp_academic_dataset_review.json':
                possible_paths.append(os.path.join(root, fname))

    review_file = None
    for p in possible_paths:
        if os.path.exists(p):
            review_file = p
            break

    if review_file is None:
        raise FileNotFoundError(
            f"Yelp review JSON not found in {data_dir}.\n"
            "Download from: https://business.yelp.com/data/resources/open-dataset/\n"
            "Click 'Download JSON', then extract the archive into data/yelp2018/\n"
            "You should have: data/yelp2018/yelp_academic_dataset_review.json"
        )

    user_map_str = {}  # string ID -> int
    item_map_str = {}
    user_counter = 0
    item_counter = 0

    user_items = defaultdict(set)
    print(f"Reading {review_file} (this may take a while)...")
    with open(review_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            review = json.loads(line)
            uid_str = review['user_id']
            bid_str = review['business_id']

            if uid_str not in user_map_str:
                user_map_str[uid_str] = user_counter
                user_counter += 1
            if bid_str not in item_map_str:
                item_map_str[bid_str] = item_counter
                item_counter += 1

            user_items[user_map_str[uid_str]].add(item_map_str[bid_str])

            if (i + 1) % 1000000 == 0:
                print(f"  Processed {i+1} reviews...")

    user_items = {u: list(items) for u, items in user_items.items()}
    print(f"Loaded Yelp: {len(user_items)} users, {len(item_map_str)} businesses")
    return user_items


def load_amazon_book(data_dir):
    """Load Amazon Books from ratings-only CSV or 5-core JSON."""
    # Try ratings CSV first (item,user,rating,timestamp)
    csv_file = os.path.join(data_dir, 'Books_ratings.csv')
    json_gz_file = os.path.join(data_dir, 'Books_5.json.gz')
    json_file = os.path.join(data_dir, 'Books_5.json')

    if os.path.exists(csv_file):
        print(f"Reading {csv_file} (this may take a while)...")
        user_map_str = {}
        item_map_str = {}
        user_counter = 0
        item_counter = 0
        user_items = defaultdict(set)

        with open(csv_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                item_str = parts[0]
                user_str = parts[1]

                if user_str not in user_map_str:
                    user_map_str[user_str] = user_counter
                    user_counter += 1
                if item_str not in item_map_str:
                    item_map_str[item_str] = item_counter
                    item_counter += 1

                user_items[user_map_str[user_str]].add(item_map_str[item_str])

                if (i + 1) % 5000000 == 0:
                    print(f"  Processed {i+1} ratings...")

        user_items = {u: list(items) for u, items in user_items.items()}
        print(f"Loaded Amazon-Book: {len(user_items)} users, {len(item_map_str)} items")
        return user_items

    elif os.path.exists(json_gz_file):
        print(f"Reading {json_gz_file} (this may take a while)...")
        user_map_str = {}
        item_map_str = {}
        user_counter = 0
        item_counter = 0
        user_items = defaultdict(set)

        with gzip.open(json_gz_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                review = json.loads(line)
                user_str = review['reviewerID']
                item_str = review['asin']

                if user_str not in user_map_str:
                    user_map_str[user_str] = user_counter
                    user_counter += 1
                if item_str not in item_map_str:
                    item_map_str[item_str] = item_counter
                    item_counter += 1

                user_items[user_map_str[user_str]].add(item_map_str[item_str])

                if (i + 1) % 5000000 == 0:
                    print(f"  Processed {i+1} reviews...")

        user_items = {u: list(items) for u, items in user_items.items()}
        print(f"Loaded Amazon-Book: {len(user_items)} users, {len(item_map_str)} items")
        return user_items

    elif os.path.exists(json_file):
        print(f"Reading {json_file} (this may take a while)...")
        user_map_str = {}
        item_map_str = {}
        user_counter = 0
        item_counter = 0
        user_items = defaultdict(set)

        with open(json_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                review = json.loads(line)
                user_str = review['reviewerID']
                item_str = review['asin']

                if user_str not in user_map_str:
                    user_map_str[user_str] = user_counter
                    user_counter += 1
                if item_str not in item_map_str:
                    item_map_str[item_str] = item_counter
                    item_counter += 1

                user_items[user_map_str[user_str]].add(item_map_str[item_str])

                if (i + 1) % 5000000 == 0:
                    print(f"  Processed {i+1} reviews...")

        user_items = {u: list(items) for u, items in user_items.items()}
        print(f"Loaded Amazon-Book: {len(user_items)} users, {len(item_map_str)} items")
        return user_items

    else:
        raise FileNotFoundError(
            f"Amazon-Book data not found in {data_dir}.\n"
            "Download from: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/\n"
            "You need one of:\n"
            "  - Books_ratings.csv (ratings only, ~2.5GB)\n"
            "  - Books_5.json.gz (5-core reviews)\n"
            "Place it in data/amazon-book/"
        )


# ============================================================
# Main
# ============================================================

DATASETS = {
    'movielens': load_movielens,
    'gowalla': load_gowalla,
    'yelp2018': load_yelp2018,
    'amazon-book': load_amazon_book,
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for LightGCN format")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()) + ['all'],
                        help="Which dataset to preprocess")
    parser.add_argument('--k-core', type=int, default=K_CORE,
                        help=f"K-core filtering threshold (default: {K_CORE})")
    parser.add_argument('--test-ratio', type=float, default=TEST_RATIO,
                        help=f"Test split ratio (default: {TEST_RATIO})")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == 'all':
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = [args.dataset]

    for dataset_name in datasets_to_process:
        data_dir = os.path.join(base_dir, dataset_name)
        loader = DATASETS[dataset_name]
        try:
            user_items = loader(data_dir)
            save_dataset(user_items, data_dir, dataset_name, k_core=args.k_core, test_ratio=args.test_ratio)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {dataset_name}: {e}")
        except Exception as e:
            print(f"\n[ERROR] {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
