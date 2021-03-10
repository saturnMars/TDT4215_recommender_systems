import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from ExplicitMF import ExplicitMF as ExplicitMatrixFactorization
import numpy as np

# Useful libraries for RS:
# a) sklearn.feature_extraction (e.g. TfidfVectorizer)
# b) sklearn.metrics.pairwise (e.g. cosine_similairty)


def MF_grid_search(train_data, test_data):
    """
    Function that test different hyperparameters for the method of 'Explicit Matrix Factorization'.
    The parameters tested are:
        a) Latent factors
        b) Regularization terms
        c) Learning rates

    :param train_data: ratings used to train the model
    :param test_data:  ratings used to evaluate the model
    :return: The best model, n_factors, reg, n_iter
    """
    latent_factors = [5, 10, 20, 40, 80]
    regularization = [1e-3, 1e-2, 1e-1, 1., 10]
    learning_rates = [1e-3, 1e-2, 1e-1]
    iter_array = [1, 2, 5, 10, 25, 50, 100]  # 200

    # Initialize best parameters
    best_params = {
        'n_factors': latent_factors[0],
        'reg': regularization[0],
        'learning_rate': None,
        'n_iter': 0,
        'train_mse': np.inf,
        'test_mse': np.inf,
        'model': None
    }

    print("Start Grid search for Explicit Matrix Factorization")
    for fact in latent_factors:
        print(f'Factors: {fact}')
        for reg in regularization:
            print(f'Regularization: {reg}')
            for rate in learning_rates:
                print(f"Rate:{rate}")

                # Define the model
                model = ExplicitMatrixFactorization(train_data, learning_alg="sgd",
                                                    n_factors=fact,
                                                    user_reg=reg, item_reg=reg,
                                                    user_bias_reg=reg, item_bias_reg=reg)
                # Test it out
                model.calculate_learning_curve(iter_array, test_data, learning_rate=rate)
                min_idx = np.argmin(model.test_mse)

                # Check if the best results (MSE on test data) have been improved
                if model.test_mse[min_idx] < best_params['test_mse']:
                    best_params['n_factors'] = fact
                    best_params['learning_rate'] = rate
                    best_params['reg'] = reg
                    best_params['n_iter'] = iter_array[min_idx]
                    best_params['train_mse'] = model.train_mse[min_idx]
                    best_params['test_mse'] = model.test_mse[min_idx]
                    best_params['model'] = model

                    print(f'--> New optimal hyperparameters')
                    print(best_params)
    return best_params["model"], (best_params['n_factors'], best_params['reg'], best_params['n_iter'])


def collaborative_filtering(train_data, test_data, find_best_model=True):
    if find_best_model:
        EF_model, n_factors, reg_term, n_iter = MF_grid_search(train_data, test_data)
        print(f"Best regularization term: {reg_term}")
        print(f"Best latent factors: {n_factors}")
        print(f"Best number of iterations: {n_iter}")
    else:
        reg = 0.0
        EF_model = ExplicitMatrixFactorization(train_data, n_factors=40,
                                               item_reg=reg, user_reg=reg,
                                               item_bias_reg=reg, user_bias_reg=reg)

    iter_array = [1, 2, 5, 10, 25, 50, 100]

    # Plot
    EF_model.plot_learning_curve(iter_array, EF_model)

    prediction = EF_model.predict_all()
    print(prediction)

    return EF_model


if __name__ == "__main__":
    # Import dataset
    df = utils.import_dataset("./data")
    print("EXAMPLE of an event:\n", df.loc[1000, :])

    # Create the User-Item matrix
    ratings = utils.Dataframe2UserItemMatrix(df)

    # Split the dataset into training and test sets
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=99)

    # METHOD 1: Model-based collaborative filtering
    # CF Latent factors: Explicit Matrix Factorization
    print("\n1) Recommendation based on the CF method (Matrix Factorization) ...")
    model_cf = collaborative_filtering(train_ratings, test_ratings)
    print("\nPREDICTIONS \n", model_cf.predictions)





    # TODO tackle the overfittig phenomenon
    # ISSUE: Huge difference between training and test data)
    # HOW: Try to follow this guide (from "Evaluation and Tuning")
    # - https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/
    # - NB: sklearn.model_selection.GridSearchCV could be useful
    # - Could be useful SGD: https://scikit-learn.org/stable/modules/sgd.html

    # ALTERNATIVE: Implement Matrix factorization through a specific library
    # https://pypi.org/project/matrix-factorization/
