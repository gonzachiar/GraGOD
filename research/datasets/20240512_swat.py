# %%

import os
import sys

import pandas as pd

day = "29"
month = "June"
year = "2020"
date = f"{day}{month}{year}"
path_to_dataset = "../../datasets_files/swat"
file = os.path.join(path_to_dataset, f"{date}.csv")
df_swat = pd.read_csv(file, sep=",")
df_swat_temp = pd.read_excel("../../datasets_files/swat/29June2020 (1).xlsx")

# %%
df_swat
# %%
df_swat_temp = pd.read_excel("../../datasets_files/swat/29June2020 (1).xlsx")

# %%
