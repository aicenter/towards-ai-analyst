# %%

import sys
sys.path.append("../..")

import math
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import polars as pl

from src import DATA_DIR

# %%

######################################################################################
###                               Data preprocessing                               ###
######################################################################################

zip_file = DATA_DIR / "amaretto_dataset_anon.csv.zip"

df = pl.read_csv(ZipFile(zip_file).read("amaretto_dataset_anon.csv"))

df = (
    df.with_columns(pl.col("Product ISIN").str.split("-").alias("product"))
    .with_columns(
        pl.col("product").list.get(0).alias("product0"),
        pl.col("product").list.get(1).alias("product1"),
        pl.col("product").list.get(2).alias("product2"),
    )
    .drop("product")
)

df = (
    df.with_row_index("id")
    .with_columns(pl.col("EntryDate").str.strptime(pl.Datetime, "%F %T", strict=False).alias("datetime"))
    .drop("EntryDate", "product0", "product1", "product2", "Product ISIN", "Originator_ID")
    .sort("datetime")
    .drop("Transaction ID")
)


df = df.with_columns(
    pl.col("Market").str.tail(1).cast(pl.Int64),
    pl.col("Currency").str.tail(1).cast(pl.Int64),
)

df.write_parquet(DATA_DIR / "amaretto.pq")

# %%

######################################################################################
###                               Feature engineering                              ###
######################################################################################

df = pl.read_parquet(DATA_DIR / "amaretto.pq")
df = df.with_columns(pl.lit(1).alias("ones"))

df = df.lazy()


index_columns = ["originator_index", "market_index", "product_type_index", "product_class_index"]


df = df.with_columns(
    pl.arange(0, pl.col("id").len()).over("Originator").alias("originator_index"),
    pl.arange(0, pl.col("id").len()).over("Market").alias("market_index"),
    pl.arange(0, pl.col("id").len()).over("Product Type").alias("product_type_index"),
    pl.arange(0, pl.col("id").len()).over("Product Class").alias("product_class_index"),
)


df = df.rename({"Normalized Amount": "amount"})


df = df.with_columns(
    (pl.col("amount") == pl.col("amount").round()).alias("is_rounded"),
)


df = df.with_columns(
    *[pl.col("datetime").dt.timestamp().diff(n=n).over("originator_index", "InputOutput").alias(f"originator_index_timestamp_diff_{n}") for n in range(1, 6)],
    *[pl.col("datetime").dt.timestamp().diff(n=n).over("market_index", "InputOutput").alias(f"market_index_timestamp_diff_{n}") for n in range(1, 6)],
    *[pl.col("datetime").dt.timestamp().diff(n=n).over("product_type_index", "InputOutput").alias(f"product_type_index_timestamp_diff_{n}") for n in range(1, 6)],
    *[pl.col("datetime").dt.timestamp().diff(n=n).over("product_class_index", "InputOutput").alias(f"product_class_index_timestamp_diff_{n}") for n in range(1, 6)],
    pl.col("amount").diff(n=1).over("originator_index", "InputOutput").alias("originator_index_amount_diff_1"),
    pl.col("amount").diff(n=1).over("market_index", "InputOutput").alias("market_index_amount_diff_1"),
    pl.col("amount").diff(n=1).over("product_type_index", "InputOutput").alias("product_type_index_amount_diff_1"),
    pl.col("amount").diff(n=1).over("product_class_index", "InputOutput").alias("product_class_index_amount_diff_1"),
)

df = df.with_columns(
    pl.col("datetime").dt.weekday().alias("day_of_week"),
    pl.col("datetime").dt.day().alias("day_of_month"),
    pl.col("datetime").dt.hour().alias("hour"),
    pl.col("datetime").dt.minute().alias("minute"),
    pl.col("datetime").dt.month_end().dt.day().alias("days_in_month"),
).with_columns(
    ((1 - pl.col("day_of_week")) * 2 * math.pi / 7).sin().name.suffix("_sin"),
    ((1 - pl.col("day_of_week")) * 2 * math.pi / 7).cos().name.suffix("_cos"),
    ((1 - pl.col("day_of_month")) * 2 * math.pi / pl.col("days_in_month")).sin().name.suffix("_sin"),
    ((1 - pl.col("day_of_month")) * 2 * math.pi / pl.col("days_in_month")).cos().name.suffix("_cos"),
    ((1 - pl.col("hour")) * 2 * math.pi / 24).sin().name.suffix("_sin"),
    ((1 - pl.col("hour")) * 2 * math.pi / 24).cos().name.suffix("_cos"),
    ((1 - pl.col("minute")) * 2 * math.pi / 60).sin().name.suffix("_sin"),
    ((1 - pl.col("minute")) * 2 * math.pi / 60).cos().name.suffix("_cos"),
)


# ### Rolling-window features
window_sizes = ["1h", "12h", "24h", "7d"]


# Global rolling counts, rolling means, and rolling std.
df = df.with_columns(
    *[pl.col("amount").rolling_mean_by("datetime", window_size=window_size, closed="left").alias(f"rolling_mean_{window_size}") for window_size in window_sizes],
    *[pl.col("amount").rolling_sum_by("datetime", window_size=window_size, closed="left").alias(f"rolling_sum_{window_size}") for window_size in window_sizes],
    *[pl.col("amount").rolling_std_by("datetime", window_size=window_size, closed="left").fill_null(0).alias(f"rolling_std_{window_size}") for window_size in window_sizes],
    *[pl.col("amount").rolling_max_by("datetime", window_size=window_size, closed="left").fill_null(0).alias(f"rolling_max_{window_size}") for window_size in window_sizes],
    *[pl.col("amount").rolling_min_by("datetime", window_size=window_size, closed="left").fill_null(0).alias(f"rolling_min_{window_size}") for window_size in window_sizes],
    *[pl.col("ones").rolling_sum_by("datetime", window_size=window_size, closed="right").alias(f"rolling_n_transactions_{window_size}") for window_size in window_sizes],
)

# Rolling amounts over customer, card, and merchant - mean and std.
df = df.with_columns(
    # Rolling mean, std, max, and min by different groupings
    # customer
    *[
        pl.col("amount").rolling_mean_by("datetime", window_size=window_size, closed="left").over("Originator").alias(f"rolling_mean_originator_{window_size}").fill_null(0)
        for window_size in window_sizes
    ],
    *[
        pl.col("amount").rolling_std_by("datetime", window_size=window_size, closed="left").over("Originator").alias(f"rolling_std_originator_{window_size}").fill_null(0)
        for window_size in window_sizes
    ],
    *[
        pl.col("amount").rolling_max_by("datetime", window_size=window_size, closed="left").over("Originator").alias(f"rolling_max_originator_{window_size}").fill_null(0)
        for window_size in window_sizes
    ],
    *[
        pl.col("amount").rolling_min_by("datetime", window_size=window_size, closed="left").over("Originator").alias(f"rolling_min_originator_{window_size}").fill_null(0)
        for window_size in window_sizes
    ],
    *[
        pl.col("amount").rolling_sum_by("datetime", window_size=window_size, closed="left").over("Originator").alias(f"rolling_sum_originator_{window_size}").fill_null(0)
        for window_size in window_sizes
    ],
)

# Rolling number of transactions per customer, card, and merchant.
df = df.with_columns(
    # customer
    *[pl.col("ones").rolling_sum_by("datetime", window_size=window_size, closed="right").over("Originator").alias(f"rolling_n_transactions_originator_{window_size}") for window_size in window_sizes],
)


cat_columns = ["InputOutput", "Market", "Product Type", "Product Class", "Currency"]
df = df.with_columns([pl.col(col).cum_count().over(col, "Originator").alias(f"cumcount_{col}") for col in cat_columns]).with_columns(
    *[(pl.col(f"cumcount_{col}") / (1 + pl.col("originator_index"))).alias(f"{col}_freq_originator") for col in cat_columns],
)
df = df.with_columns(np.log1p(pl.col("amount")).alias("amount_log"))

prior_columns_1 = [
    "id",
    "Originator",
    "Anomaly",
    "day_of_week",
    "day_of_month",
    "hour",
    "minute",
    "amount_log",
    "InputOutput", "Market", "Product Type", "Product Class", "Currency",
]
prior_columns_2 = [
    "InputOutput_Buy",
    "InputOutput_Sell",
    "Market_1",
    "Market_2",
    "Market_3",
    "Market_4",
    "Product Type_ADR",
    "Product Type_Bond",
    "Product Type_CAADR",
    "Product Type_ETOEquity",
    "Product Type_ETOEquityIndex",
    "Product Type_Equity",
    "Product Type_FX",
    "Product Type_FXForward",
    "Product Type_FXSwap",
    "Product Type_FutureBond",
    "Product Type_FutureCommodity",
    "Product Type_FutureEquity",
    "Product Type_FutureEquityIndex",
    "Product Type_FutureFX",
    "Product Type_FutureOptionEquityIndex",
    "Product Type_Repo",
    "Product Type_SimpleTransfer",
    "Product Class_ADR Conversion",
    "Product Class_Cash in / out (withdrawal), Security in / out",
    "Product Class_External fee",
    "Product Class_Trade",
    "Currency_1",
    "Currency_2",
]
prior_columns = [
    "id",
    "Originator",
    "Anomaly",
    "InputOutput_Buy",
    "InputOutput_Sell",
    "Market_1",
    "Market_2",
    "Market_3",
    "Market_4",
    "Product Type_ADR",
    "Product Type_Bond",
    "Product Type_CAADR",
    "Product Type_ETOEquity",
    "Product Type_ETOEquityIndex",
    "Product Type_Equity",
    "Product Type_FX",
    "Product Type_FXForward",
    "Product Type_FXSwap",
    "Product Type_FutureBond",
    "Product Type_FutureCommodity",
    "Product Type_FutureEquity",
    "Product Type_FutureEquityIndex",
    "Product Type_FutureFX",
    "Product Type_FutureOptionEquityIndex",
    "Product Type_Repo",
    "Product Type_SimpleTransfer",
    "Product Class_ADR Conversion",
    "Product Class_Cash in / out (withdrawal), Security in / out",
    "Product Class_External fee",
    "Product Class_Trade",
    "Currency_1",
    "Currency_2",
    "day_of_week",
    "day_of_month",
    "hour",
    "minute",
    "amount_log",
]

costly_features_columns = [
    "id",
    "Originator",
    "Anomaly",
    "is_rounded",
    # np.log1p(pl.col('originator_index_timestamp_diff_0').fill_null(0)),
    np.log1p(pl.col("originator_index_timestamp_diff_1").fill_null(0)),
    np.log1p(pl.col("originator_index_timestamp_diff_2").fill_null(0)),
    np.log1p(pl.col("originator_index_timestamp_diff_3").fill_null(0)),
    np.log1p(pl.col("originator_index_timestamp_diff_4").fill_null(0)),
    np.log1p(pl.col("originator_index_timestamp_diff_5").fill_null(0)),
    # np.log1p(pl.col('market_index_timestamp_diff_0').fill_null(0)),
    np.log1p(pl.col("market_index_timestamp_diff_1").fill_null(0)),
    np.log1p(pl.col("market_index_timestamp_diff_2").fill_null(0)),
    np.log1p(pl.col("market_index_timestamp_diff_3").fill_null(0)),
    np.log1p(pl.col("market_index_timestamp_diff_4").fill_null(0)),
    np.log1p(pl.col("market_index_timestamp_diff_5").fill_null(0)),
    # np.log1p(pl.col('product_type_index_timestamp_diff_0').fill_null(0)),
    np.log1p(pl.col("product_type_index_timestamp_diff_1").fill_null(0)),
    np.log1p(pl.col("product_type_index_timestamp_diff_2").fill_null(0)),
    np.log1p(pl.col("product_type_index_timestamp_diff_3").fill_null(0)),
    np.log1p(pl.col("product_type_index_timestamp_diff_4").fill_null(0)),
    np.log1p(pl.col("product_type_index_timestamp_diff_5").fill_null(0)),
    # np.log1p(pl.col('product_class_index_timestamp_diff_0').fill_null(0)),
    np.log1p(pl.col("product_class_index_timestamp_diff_1").fill_null(0)),
    np.log1p(pl.col("product_class_index_timestamp_diff_2").fill_null(0)),
    np.log1p(pl.col("product_class_index_timestamp_diff_3").fill_null(0)),
    np.log1p(pl.col("product_class_index_timestamp_diff_4").fill_null(0)),
    np.log1p(pl.col("product_class_index_timestamp_diff_5").fill_null(0)),
    np.sign(pl.col("originator_index_amount_diff_1").fill_null(0)) * np.log1p(pl.col("originator_index_amount_diff_1").fill_null(0).abs()),
    np.sign(pl.col("market_index_amount_diff_1").fill_null(0)) * np.log1p(pl.col("market_index_amount_diff_1").fill_null(0).abs()),
    np.sign(pl.col("product_type_index_amount_diff_1").fill_null(0)) * np.log1p(pl.col("product_type_index_amount_diff_1").fill_null(0).abs()),
    np.sign(pl.col("product_class_index_amount_diff_1").fill_null(0)) * np.log1p(pl.col("product_class_index_amount_diff_1").fill_null(0).abs()),
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    # # 'ones',
    np.log1p(pl.col("rolling_mean_1h").fill_null(0)),
    np.log1p(pl.col("rolling_mean_12h").fill_null(0)),
    np.log1p(pl.col("rolling_mean_24h").fill_null(0)),
    np.log1p(pl.col("rolling_mean_7d").fill_null(0)),
    np.log1p(pl.col("rolling_sum_1h").fill_null(0)),
    np.log1p(pl.col("rolling_sum_12h").fill_null(0)),
    np.log1p(pl.col("rolling_sum_24h").fill_null(0)),
    np.log1p(pl.col("rolling_sum_7d").fill_null(0)),
    np.log1p(pl.col("rolling_std_1h").fill_null(0)),
    np.log1p(pl.col("rolling_std_12h").fill_null(0)),
    np.log1p(pl.col("rolling_std_24h").fill_null(0)),
    np.log1p(pl.col("rolling_std_7d").fill_null(0)),
    np.log1p(pl.col("rolling_max_1h").fill_null(0)),
    np.log1p(pl.col("rolling_max_12h").fill_null(0)),
    np.log1p(pl.col("rolling_max_24h").fill_null(0)),
    np.log1p(pl.col("rolling_max_7d").fill_null(0)),
    np.log1p(pl.col("rolling_min_1h").fill_null(0)),
    np.log1p(pl.col("rolling_min_12h").fill_null(0)),
    np.log1p(pl.col("rolling_min_24h").fill_null(0)),
    np.log1p(pl.col("rolling_min_7d").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_1h").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_12h").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_24h").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_7d").fill_null(0)),
    np.log1p(pl.col("rolling_mean_originator_1h").fill_null(0)),
    np.log1p(pl.col("rolling_mean_originator_12h").fill_null(0)),
    np.log1p(pl.col("rolling_mean_originator_24h").fill_null(0)),
    np.log1p(pl.col("rolling_mean_originator_7d").fill_null(0)),
    np.log1p(pl.col("rolling_std_originator_1h").fill_null(0)),
    np.log1p(pl.col("rolling_std_originator_12h").fill_null(0)),
    np.log1p(pl.col("rolling_std_originator_24h").fill_null(0)),
    np.log1p(pl.col("rolling_std_originator_7d").fill_null(0)),
    np.log1p(pl.col("rolling_max_originator_1h").fill_null(0)),
    np.log1p(pl.col("rolling_max_originator_12h").fill_null(0)),
    np.log1p(pl.col("rolling_max_originator_24h").fill_null(0)),
    np.log1p(pl.col("rolling_max_originator_7d").fill_null(0)),
    np.log1p(pl.col("rolling_min_originator_1h").fill_null(0)),
    np.log1p(pl.col("rolling_min_originator_12h").fill_null(0)),
    np.log1p(pl.col("rolling_min_originator_24h").fill_null(0)),
    np.log1p(pl.col("rolling_min_originator_7d").fill_null(0)),
    np.log1p(pl.col("rolling_sum_originator_1h").fill_null(0)),
    np.log1p(pl.col("rolling_sum_originator_12h").fill_null(0)),
    np.log1p(pl.col("rolling_sum_originator_24h").fill_null(0)),
    np.log1p(pl.col("rolling_sum_originator_7d").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_originator_1h").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_originator_12h").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_originator_24h").fill_null(0)),
    np.log1p(pl.col("rolling_n_transactions_originator_7d").fill_null(0)),
    np.log1p(pl.col("cumcount_InputOutput").fill_null(0)),
    np.log1p(pl.col("cumcount_Market").fill_null(0)),
    np.log1p(pl.col("cumcount_Product Type").fill_null(0)),
    np.log1p(pl.col("cumcount_Product Class").fill_null(0)),
    np.log1p(pl.col("cumcount_Currency").fill_null(0)),
    np.log1p(pl.col("InputOutput_freq_originator").fill_null(0)),
    np.log1p(pl.col("Market_freq_originator").fill_null(0)),
    np.log1p(pl.col("Product Type_freq_originator").fill_null(0)),
    np.log1p(pl.col("Product Class_freq_originator").fill_null(0)),
    np.log1p(pl.col("Currency_freq_originator").fill_null(0)),
]

######################################################################################
###                    Write prior and costly features parquets                    ###
######################################################################################

amaretto_path = DATA_DIR / 'amaretto'
amaretto_path.mkdir(exist_ok=True)

# create time split into train, validation, test sets
df = df.with_columns(pl.when(pl.col("datetime") < datetime(2019, 3, 1)).then(pl.lit(0)).when(pl.col("datetime") > datetime(2019, 3, 15)).then(pl.lit(2)).otherwise(pl.lit(1)).alias("split"))

for i, split in enumerate(["train", "val", "test"]):
    indexes = df.filter(pl.col("split") == i).select("id").collect()["id"].to_list()

    df_prior = (
        df
        .filter(pl.col("id").is_in(indexes))
        .sort("id")
        .select(prior_columns_1)
        .collect()
        .to_dummies(["InputOutput", "Market", "Product Type", "Product Class", "Currency"])
        .select(prior_columns)
        .write_parquet(DATA_DIR / f"amaretto/{split}_prior.pq")
    )
    df.filter(pl.col("id").is_in(indexes)).sort("id").select(costly_features_columns).collect().write_parquet(DATA_DIR / f"amaretto/{split}_costly_features.pq")


df.select(prior_columns_1).collect().to_dummies(["InputOutput", "Market", "Product Type", "Product Class", "Currency"]).select(prior_columns).write_parquet(DATA_DIR / "amaretto/prior.pq")
df.select(costly_features_columns).sink_parquet(DATA_DIR / "amaretto/costly_features.pq")

# %%
