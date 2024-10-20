# %%
import pandas as pd

from gragod import start_research

DATASET_PATH = "datasets_files/dataset_end_preprocessed.csv"
SERVER_4_PATH = "datasets_files/server_4.csv"
# %%

df = pd.read_csv(DATASET_PATH)
# %%
df.head()
# Which datasets is this one?
df.shape
# %%
(len(df) / (24 * 60 * 60)) ** (-1)
# %%
from enum import Enum
from typing import Literal


class AnomalyType(Enum):
    cpu_overload = "cpu_overload"
    memory_overload = "memory_overload"
    disk_overload = "disk_overload"


TEST_5G3E_PATH = "datasets_files/5G3E/Test_Set_Day/{anomaly_type}/{variation}/{level}/server_{n_server}_test_{n_test}.csv"
df_1_1 = pd.read_csv(
    TEST_5G3E_PATH.format(
        anomaly_type=AnomalyType.cpu_overload.value,
        variation="20",
        level="physical_level",
        n_server=1,
        n_test=1,
    ),
    delimiter=";",
    on_bad_lines="skip",
)
df_1_2 = pd.read_csv(
    TEST_5G3E_PATH.format(
        anomaly_type=AnomalyType.cpu_overload.value,
        variation="20",
        level="physical_level",
        n_server=1,
        n_test=2,
    ),
    delimiter=";",
    on_bad_lines="skip",
)
df_2_1 = pd.read_csv(
    TEST_5G3E_PATH.format(
        anomaly_type=AnomalyType.cpu_overload.value,
        variation="20",
        level="physical_level",
        n_server=2,
        n_test=1,
    ),
    delimiter=";",
    on_bad_lines="skip",
)
df_2_2 = pd.read_csv(
    TEST_5G3E_PATH.format(
        anomaly_type=AnomalyType.cpu_overload.value,
        variation="20",
        level="physical_level",
        n_server=2,
        n_test=2,
    ),
    delimiter=";",
    on_bad_lines="skip",
)
df_1_1
# %%
df_1_1.dropna(axis=1, inplace=True)
df_1_2.dropna(axis=1, inplace=True)
df_2_1.dropna(axis=1, inplace=True)
df_2_2.dropna(axis=1, inplace=True)
df_1_1
# %%
df_1_train = pd.read_csv(
    "datasets_files/5G3E/Training_Set_Day_1/physical_level/server_1.csv", delimiter=";"
)
df_2_train = pd.read_csv(
    "datasets_files/5G3E/Training_Set_Day_1/physical_level/server_2.csv", delimiter=";"
)
# %%
df_1_train
# %%
cols_train = set(df_1_train.columns)
cols_1_1 = set(df_1_1.columns)

# %%
# See different columns
cols_train - cols_1_1
# %%
df_1_train['promhttp_metric_handler_requests_total{code="500"}'].value_counts()
df_1_1['promhttp_metric_handler_requests_total{code="500"}'].value_counts()
# %%
df_1_train
# %%
df_2_train
# %%
