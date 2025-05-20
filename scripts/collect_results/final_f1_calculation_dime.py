import argparse
import gc
import os

import numpy as np
import polars as pl
from sklearn.metrics import f1_score
from tqdm import tqdm

from src import RESULTS_DIR
from src.datasets import AmarettoDataset
from utils.evaluation import f1_smart


def results_confidence_threshold(confidence_threshold: float, confidences, probabilities):
    confidence_thr = []
    prob_thr = []
    n_features_thr = []

    for i, sample in enumerate(confidences.transpose()):
        for ix, feature_confidence in enumerate(sample):
            conf = sample[ix]
            if conf > confidence_threshold:
                confidence_thr.append(conf)
                prob_thr.append(probabilities[ix, i])
                n_features_thr.append(ix)
                break

        else:
            confidence_thr.append(conf)
            prob_thr.append(probabilities[ix, i])
            n_features_thr.append(ix)

    return confidence_thr, prob_thr, n_features_thr


def results_confidence_drop(confidences, probabilities):
    confidence_before_drop = []
    prob_before_drop = []
    n_features_before_drop = []

    for i, sample in enumerate(confidences.transpose()):
        max_conf = 0

        for ix, feature_confidence in enumerate(sample):
            if feature_confidence >= max_conf:
                max_conf = feature_confidence
            else:
                confidence_before_drop.append(max_conf)
                prob_before_drop.append(probabilities[ix - 1, i])
                n_features_before_drop.append(ix - 1)
                break

        else:
            confidence_before_drop.append(max_conf)
            n_features_before_drop.append(len(sample) - 1)
            prob_before_drop.append(probabilities[len(sample) - 1, i])

    return confidence_before_drop, prob_before_drop, n_features_before_drop


# train_dataset = AmarettoDataset(split="train")
val_dataset = AmarettoDataset(split="val")
test_dataset = AmarettoDataset(split="test")

val_labels = val_dataset.labels
test_labels = test_dataset.labels


del val_dataset, test_dataset
gc.collect()

path = RESULTS_DIR / "amaretto/dime_old"
path_list = os.listdir(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Results")
    parser.add_argument("--index", type=int, default=None, help="Index of the file.")
    return parser.parse_args()


args = parse_args()
index = args.index
p = path_list[index]

val_path = f"{path}/{p}/val.npz"
test_path = f"{path}/{p}/test.npz"

try:
    val_results = np.load(val_path)
    test_results = np.load(test_path)
except:
    print("No files for this model, skipping...")
    raise ValueError

# validation data
confidences = val_results["confidences"]
probabilities = val_results["probabilities"]
queried_features = val_results["queried_features"]
print("validation data loaded")

# test data
test_confidences = test_results["confidences"]
test_probabilities = test_results["probabilities"]
test_queried_features = test_results["queried_features"]
print("test data loaded")

# budget results
budget_results = []
n = probabilities.shape[0]

for n_features in tqdm(range(n)):
    _, thr = f1_smart(val_labels, probabilities[n_features, :])
    val_f1 = f1_score(val_labels, probabilities[n_features, :] >= thr)
    test_f1 = f1_score(test_labels, test_probabilities[n_features, :] >= thr)

    budget_results.append([val_f1, test_f1, thr])

df = pl.DataFrame(np.array(budget_results), schema=["val_f1", "test_f1", "threshold"]).with_row_index("budget").with_columns(pl.col("budget").cast(pl.Float64))
budget, max_val, max_test, thr = df.sort("val_f1", descending=True)[0]
budget, max_val, max_test, thr = budget.item(), max_val.item(), max_test.item(), thr.item()

print(f"Budget results for {budget} features:")
print("F1 validation:", round(max_val, 4))
print("F1 test:      ", round(max_test, 4))

df = df.with_columns(pl.lit("budget").alias("type"))

# confidence thresholding
confidence_threshold = 0.99
confidence_thr, prob_thr, n_features_thr = results_confidence_threshold(confidence_threshold, confidences, probabilities)
test_confidence_thr, test_prob_thr, test_n_features_thr = results_confidence_threshold(confidence_threshold, test_confidences, test_probabilities)

_, thr = f1_smart(val_labels, prob_thr)
val_f1_thr = f1_score(val_labels, prob_thr >= thr)
test_f1_thr = f1_score(test_labels, test_prob_thr >= thr)
print("Confidence threshold:", confidence_threshold)
print("Optimized threshold:", round(thr, 5))
print("F1 validation:", round(val_f1_thr, 4))
print("F1 test:      ", round(test_f1_thr, 4))

schema = ["budget", "val_f1", "test_f1", "threshold"]
df = df.vstack(pl.DataFrame(np.array([np.mean(test_n_features_thr), val_f1_thr, test_f1_thr, thr]).reshape(1, -1), schema=schema).with_columns(pl.lit("confidence_threshold").alias("type")))

# max confidence before drop
confidence_before_drop, prob_before_drop, n_features_before_drop = results_confidence_drop(confidences, probabilities)
test_confidence_before_drop, test_prob_before_drop, test_n_features_before_drop = results_confidence_drop(test_confidences, test_probabilities)

_, thr2 = f1_smart(val_labels, prob_before_drop)
val_f1_conf = f1_score(val_labels, prob_before_drop >= thr2)
test_f1_conf = f1_score(test_labels, test_prob_before_drop >= thr2)

print("Optimized threshold:", round(thr2, 5))
print("F1 validation:", round(val_f1_conf, 4))
print("F1 test:      ", round(test_f1_conf, 4))

df = df.vstack(
    pl.DataFrame(np.array([np.mean(test_n_features_before_drop), val_f1_conf, test_f1_conf, thr2]).reshape(1, -1), schema=schema).with_columns(pl.lit("confidence_before_drop").alias("type"))
)

df.write_parquet(f"{path}/{p}/f1_results.pq")
