import utils

if __name__ == "__main__":
    # Import dataset
    df = utils.import_dataset("./data")
    print("EXAMPLE:\n", df.loc[1000, :])

