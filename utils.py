from pprint import PrettyPrinter
import json
import os


def import_dataset(folder_path, full_output=False):
    files = os.listdir(folder_path)

    events_imported = 0
    files_imported = 0

    # Scan each file in the directory
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        num_events = 0

        with open(file_path) as file:

            # Retrieve all events within the file
            for line in file:
                event = json.loads(line.strip())

                # TODO Save each object (event: JSON file) into a data structure (e.g. pandas dataframe)

                num_events += 1
                if full_output:
                    PrettyPrinter(indent=4).pprint(event)

            files_imported += 1
            events_imported += num_events
            print("LOADED: {0} events for file {1}".format(num_events, file_name))

    print("Files imported: ", files_imported)
    print("Events imported: ", events_imported)

    return None
