import utils
from sklearn.model_selection import train_test_split
from ExplicitMF import ExplicitMF as ExplicitMatrixFactorization


def collaborative_filtering(ratings):
    # Split the dataset into training and test sets
    train_events, test_events = train_test_split(ratings, test_size=0.2, random_state=99)

    # Run the Algorithm of Explicit Matrix Factorization
    mf = ExplicitMatrixFactorization(train_events, verbose=True)
    predictions = mf.predictions

    # Test it on unseen data (train set)
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    mf.calculate_learning_curve(iter_array, test_events)

    # Plot
    mf.plot_learning_curve(iter_array)

    return predictions


if __name__ == "__main__":

    # Import dataset
    df = utils.import_dataset("./data")
    print("EXAMPLE of an event:\n", df.loc[1000, :])

    # Create the User-Item matrix
    ratings = utils.Dataframe2UserItemMatrix(df)

    # Method 1: Collaborative filtering through the "Explicit Matrix Factorization" algorithm
    print("\n1) Recommendation based on the CF method (Matrix Factorization) ...")
    recommendations_cf = collaborative_filtering(ratings)
    print("\nRECOMMENDATIONS \n", recommendations_cf)




