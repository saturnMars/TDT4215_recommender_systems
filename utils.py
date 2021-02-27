from pandas import read_json, concat

from os import listdir, path


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
