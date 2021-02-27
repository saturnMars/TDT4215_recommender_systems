from pandas import read_json, concat, merge, DataFrame
from os import listdir, path
import numpy as np



def import_dataset(folder_path):
    """
    Read and save as Pandas DF the events includes in all the JSON files
    :param folder_path: system path of the folder to read
    :return: Pandas DataFrame with all the events
    """
    files = listdir(folder_path)

    # Scan each file in the directory
    daily_events = list()
    for file_name in files:
        file_path = path.join(folder_path, file_name)

        # Read the file
        with open(file_path) as file:
            events = read_json(file, lines=True)
            daily_events.append(events)

            print("LOADED: {0} events for file {1}".format(len(events), file_name))

    # Concatenate all the daily events
    df_events = concat(daily_events, ignore_index=True)

    print("TOTAL EVENTS: {0}\n".format(len(df_events)))
    return df_events


def Dataframe2UserItemMatrix(df):
    """
    @author: zhanglemei and peng -  Sat Jan  5 13:48:20 2019

    Convert dataframe to user-item-interaction matrix, which is used for
    Matrix Factorization based recommendation.
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

    for row in df_ext.itertuples():
        ratings[row[1] - 1, row[2] - 1] = 1.0

    # Print ratings matrix
    print("USER-ITEM MATRIX: \n", ratings)

    # Print ratings available (1s)
    unique, counter = np.unique(ratings, return_counts=True)
    ratings_available = dict(zip(unique, counter))
    sparsity = 100 * round((ratings_available[1] / ratings_available[0]), 4)
    print(f"Number of ratings available (1s): {ratings_available[1]} (~ {sparsity} %, "
          f"total = {sum(ratings_available.values())}]")

    return ratings


