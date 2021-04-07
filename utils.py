from pandas import read_json, read_csv, concat, merge, DataFrame, Timestamp, date_range, to_datetime
from os import listdir, path
import numpy as np


def import_dataset(folder_path):
    """
    Read and save as Pandas DF the events includes in all the JSON files
    :param folder_path: system path of the folder to read
    :return: Pandas DataFrame with all the events
    """
    csv_file_path = path.join(folder_path, "Events.csv")

    # if path.isfile(csv_file_path):
    #     df = read_csv(csv_file_path)
    #     df["time"] = to_datetime(df["time"])
    #     print(f"Dataset loaded with {len(df)} events")
    #     return df

    files = listdir(folder_path)

    # Scan each file in the directory
    daily_events = list()
    for file_name in files:
        file_path = path.join(folder_path, file_name)

        # Read the file
        with open(file_path) as file:
            events = read_json(file, lines=True)

            # Convert the Unix timestamp to ordinary dates
            events["time"] = events["time"].apply(
                lambda date: Timestamp(date, unit="s", tz="Europe/Oslo")
            )
            daily_events.append(events)

            print("LOADED: {0} events for file {1}".format(len(events), file_name))

    # Concatenate all the daily events
    df_events = concat(daily_events, ignore_index=True)
    # df_events.to_csv(csv_file_path, index=False)

    print(f"TOTAL EVENTS: {len(df_events)}")
    return df_events


def split_train_test(df, num_test_days):
    last_day = df['time'].iloc[-1].date()
    test_window = date_range(end=last_day, periods=num_test_days, freq="D").date

    # Split into test and train dataset
    test_mask = df["time"].dt.date.isin(test_window)
    train_df = df[-test_mask]
    test_df = df[test_mask]

    # Find common users
    common_users = set(test_df["userId"]).intersection(train_df["userId"])
    return train_df, test_df, common_users


def Dataframe2UserItemMatrix(df, common_users):
    """
    @author: zhanglemei and peng -  Sat Jan  5 13:48:20 2019

    Convert dataframe to user-item-interaction matrix, which is used for
    Matrix Factorization based recommendation.
    ROWS: users
    COLUMNS: items
    In rating matrix, clicked events are refered as 1 and others are refered as 0.

    :param df: Pandas Dataframe
    :return: ratings in a User-Item matrix
    """
    df = df[~df['documentId'].isnull()]
    df = df.drop_duplicates(subset=['userId', 'documentId'])
    df = df.sort_values(by=['userId', 'time'])

    n_users = df['userId'].nunique()
    n_items = df['documentId'].nunique()

    ratings = np.zeros((n_users, n_items))

    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    new_user = np.r_[True, new_user]

    df['uid'] = np.cumsum(new_user)
    item_ids = df['documentId'].unique().tolist()

    new_df = DataFrame({'documentId': item_ids, 'tid': range(1, len(item_ids) + 1)})

    df = merge(df, new_df, on='documentId', how='outer')
    df_ext = df[['uid', 'tid']]

    # Find indexes of common users
    common_users_df = df[df["userId"].isin(common_users)]["uid"].unique()
    common_idx = set()
    event_idx = set()

    for row in df_ext.itertuples():
        ratings[row[1] - 1, row[2] - 1] = 1.0

        if row[1] in common_users_df:
            common_idx.add(row[1] - 1)

    # Print ratings matrix
    print(f"\nThe User-Item Matrix has been generated ({ratings.shape[0]} users and {ratings.shape[1]} items)")

    # Print ratings available (1s)
    unique, counter = np.unique(ratings, return_counts=True)
    ratings_available = dict(zip(unique, counter))
    sparsity = round(100 * (ratings_available[1] / ratings_available[0]), 2)
    print(f"Number of ratings available (1s): {ratings_available[1]} "
          f"(~ {sparsity} %, total = {sum(ratings_available.values())}]")

    return ratings, common_idx, item_ids
