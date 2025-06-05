# Towards AI Analyst: Querying Costly Features for Fraud and Money Laundering Detection

The repository shares code for the paper Towards AI Analyst: Querying Costly Features for Fraud and Money Laundering Detection.

## Structure

Source code lives in the `src` folder. All scripts are included in the `scripts` folder.

## Getting started

### Install

Install the requirements from the `requirements.txt` file. The code has been run on Python/3.12.

### Data

Public data and models live in the `local_caches` folder. You can find all the necessary data on Google Drive here: [towards-ai-analyst-data](https://drive.google.com/drive/folders/1uz7nSf-zbwMfkGqUbtGb_wCBhM3AOIA1?usp=drive_link). The structure is divided into `data` folder and `results` folder.

#### `data`

The original Amaretto dataset can be found at [https://github.com/necst/amaretto_dataset](https://github.com/necst/amaretto_dataset). Download from the `data` subfolder on the Google Drive.

- `amaretto_dataset_anon.csv.zip`: file that is just the original Amaretto dataset processed into on .csv file
- `amaretto.pq`: full dataframe with extracted features via feature engineering (as in `scripts/data_prep/feature_engineering.py`)
- `amaretto.tar.gz` that unpacks to a `amaretto` folder with
    - `prior.pq`, `val_prior.pq`, and `test_prior.pq` that are time split of *prior* features
    - `costly_features.pq`, `val_costly_features.pq`, and `test_costly_features.pq` that are time split of *costly* features


#### `results`

Results are to be unpacked in `local_caches/results` folder from the file `amaretto_results.tar.gz`. The results include calculated scores and other data from the best trained models in all settings. If you'd like to get the trained models themselves, don't hesitate to write to us and we can provide them.

```bash
❯ tar -tzf amaretto_results.tar.gz

amaretto/
amaretto/dime/
amaretto/nn_classifier_prior_probabilities.pkl
amaretto/nn_classifier_full_probabilities.pkl
amaretto/nn_classifier_2f_probabilities.pkl
amaretto/dime/test.npz
amaretto/dime/val.npz
```

## Test it yourself

You can load the data and the model results and try for yourself using the notebook `notebooks/costly_features_results.ipynb`.

## Cite

To cite the paper, please use

> [tbd]

## References

> Gadgil, S., Covert, I.C. and Lee, S.I., Estimating Conditional Mutual Information for Dynamic Feature Selection. In The Twelfth International Conference on Learning Representations.
