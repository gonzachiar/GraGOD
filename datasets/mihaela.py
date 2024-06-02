import datetime
import os
from typing import List, Literal, Optional

import pandas as pd

from datasets.data_processing import INTERPOLATION_METHODS, preprocess_df

CITIES = Literal["Nancy"]
BASE_PATH_DEFAULT = "datasets_files/mihaela"
DEFAULT_DATES = [str(day) for day in range(20190501, 20190532)]


def load_mihaela_service_df(
    service_name: str = "Clash_of_Clans",
    dates: List[str] = DEFAULT_DATES,
    base_path: str = BASE_PATH_DEFAULT,
    city_name: CITIES = "Nancy",
) -> pd.DataFrame:
    """
    Loads the service data for a given city and service.
    Args:
        service_name: The name of the service.
        dates: The list of dates to load.
        base_path: The path to the datasets.
        city_name: The name of the city.
    Returns:
        The DataFrame with the service data. In the first column is the
        is the time, in the other columns are the traffic values for each tile_id.
    """
    all_dfs = []
    for day_str in dates:
        traffic_file = os.path.join(
            base_path,
            city_name,
            service_name,
            day_str,
            f"{city_name}_{service_name}_{day_str}_DL.txt",
        )
        day = datetime.datetime.strptime(day_str, "%Y%m%d")
        times = [day + datetime.timedelta(minutes=15 * i) for i in range(96)]
        day_times_str = [str(day) + t.strftime("%H:%M") for t in times]

        column_names = ["tile_id"] + day_times_str

        df_traffic_dn = pd.read_csv(traffic_file, sep=" ", names=column_names)
        df_pivot = df_traffic_dn.melt(
            id_vars=["tile_id"], var_name="time", value_name="traffic"
        )
        df_pivot = df_pivot.pivot(
            index="time", columns="tile_id", values="traffic"
        ).reset_index()
        all_dfs.append(df_pivot)

    final_df = pd.concat(all_dfs, axis=0)
    print(final_df.columns)
    return final_df


def load_mihaela_service_training_data(
    service_name: str,
    dates: List[str] = DEFAULT_DATES,
    base_path: str = BASE_PATH_DEFAULT,
    city_name: CITIES = "Nancy",
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: Optional[INTERPOLATION_METHODS] = None,
):
    """
    Load the training data for the given service from Mihaela dataset.
    Args:
        service_name: The name of the service.
        dates: The list of dates to load.
        base_path: The path to the datasets.
        city_name: The name of the city.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
        scaler: The scaler to use for normalization.
        interpolate_method: The method to use for interpolation.
    """
    df = load_mihaela_service_df(
        service_name=service_name,
        dates=dates,
        base_path=base_path,
        city_name=city_name,
    )

    return preprocess_df(
        data_df=df,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )
