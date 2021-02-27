import utils
from sklearn.model_selection import train_test_split
from ExplicitMF import ExplicitMF as ExplicitMatrixFactorization

# Useful libraries for RS:
# a) sklearn.feature_extraction (e.g. TfidfVectorizer)
# b) sklearn.metrics.pairwise (e.g. cosine_similairty)


def collaborative_filtering(ratings):
    # Split the dataset into training and test sets
    train_events, test_events = train_test_split(ratings, test_size=0.2, random_state=99)

    # Run the Algorithm of Explicit Matrix Factorization
    model = ExplicitMatrixFactorization(train_events, n_factors=40, item_reg=1.0, user_reg=1.0,
                                        verbose=True)

    # Test it on unseen data (train set)
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    model.calculate_learning_curve(iter_array, test_events)

    # Plot
    model.plot_learning_curve(iter_array)
    return model


if __name__ == "__main__":
    # Import dataset
    df = utils.import_dataset("./data")
    print("EXAMPLE of an event:\n", df.loc[1000, :])

    # Create the User-Item matrix
    ratings = utils.Dataframe2UserItemMatrix(df)

    # METHOD 1: Model-based collaborative filtering
    # Latent factors: Explicit Matrix Factorization
    print("\n1) Recommendation based on the CF method (Matrix Factorization) ...")
    model_cf = collaborative_filtering(ratings)
    print("\nPREDICTIONS \n", model_cf.predictions)

    # TODO tackle the overfittig phenomenon
    # ISSUE: Huge difference between training and test data)
    # HOW: Try to follow this guide (from "Evaluation and Tuning")
    # - https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/
    # - NB: sklearn.model_selection.GridSearchCV could be useful
    # - Could be useful SGD: https://scikit-learn.org/stable/modules/sgd.html

    # ALTERNATIVE: Implement Matrix factorization through a specific library
    # https://pypi.org/project/matrix-factorization/
