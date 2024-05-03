import datetime
import os

import pandas as pd


def load_service(
    service_name: str,
    dates: list[str] = [str(day) for day in range(20190501, 20190532)],
    path_to_datasets: str = "../datasets_files/mihaela",
    city_name: str = "Nancy",
) -> pd.DataFrame:
    """
    Loads the service data for a given city and service.
    Args:
        service_name: The name of the service.
        dates: The list of dates to load.
        path_to_datasets: The path to the datasets.
        city_name: The name of the city.
    Returns:
        The DataFrame with the service data. In the first column is the
        is the time, in the other columns are the traffic values for each tile_id.
    """
    all_dfs = []
    for day_str in dates:
        traffic_file = os.path.join(
            path_to_datasets,
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
