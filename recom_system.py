import utils
from ExplicitMF import ExplicitMF as ExplicitMatrixFactorization
import numpy as np

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
    regularization = [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    iter_array = [1, 2, 5, 10, 25, 50, 100]

    # Initialize best parameters
    best_params = {
        'n_factors': latent_factors[0],
        'reg': regularization[0],
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
            # Define the model
            model = ExplicitMatrixFactorization(train_data, learning_alg="sgd",
                                                n_factors=fact,
                                                user_reg=reg, item_reg=reg,
                                                user_bias_reg=reg, item_bias_reg=reg)
            # Test it out
            model.calculate_learning_curve(iter_array, test_data)
            min_idx = np.argmin(model.test_mse)

            # Check if the best results (MSE on test data) have been improved
            if model.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = model.train_mse[min_idx]
                best_params['test_mse'] = model.test_mse[min_idx]
                best_params['model'] = model

                print(f'--> New optimal hyperparameters')
                print(best_params)

    return best_params["model"], (best_params['n_factors'], best_params['reg'], best_params['n_iter'])


def collaborative_filtering(train_data, test_data, find_best_model=False):
    """
    Implement the algorithm of Explicit Matrix factorization using an approach of collaborative filtering
    :param train_data: data used to train the model
    :param test_data: data used to evaluate the model
    :param find_best_model: (bool) Set true to try to find the best hyperparameter for the algorithm
                                   it should be set to TRUE just the first time for finding the best value
                                   to initialize the model

    :return: Prediction on train data, MSE on test data
    """
    if find_best_model:
        EF_model, n_factors, reg_term, n_iter = MF_grid_search(train_data, test_data)

        # EF_model.plot_learning_curve(iter_array, EF_model)

        print(f"Best regularization term: {reg_term}")
        print(f"Best latent factors: {n_factors}")
        print(f"Best number of iterations: {n_iter}")

        # New optimal hyperparameters ('n_factors': 5, 'reg': 1.0, 'n_iter': 100)
        # 'train_mse': 4.714649845685047e-26,
        # 'test_mse': 2.957891863472903e-26

        # n_iter 1: 3min, mse = e-06
        # n_iter 5: 5min, mse = e-07
        # n_iter 10: 7min, mse = e-09
        # n_iter 25: 15min, mse = e-11
        # n_iter 50: 25min, mse = e-17
        # n_iter 100: 1.30h, mse = e-26

    # Initialize the model with the best parameters found with grid search
    else:
        num_factors = 5
        num_iter = 5
        reg_term = 1
        learning_rate = 0.1
        EF_model = ExplicitMatrixFactorization(train_data, n_factors=num_factors, learning_alg="sgd",
                                               item_reg=reg_term, user_reg=reg_term,
                                               item_bias_reg=reg_term, user_bias_reg=reg_term)
    # Train the model
    EF_model.train(num_iter, learning_rate)

    # Prediction
    predictions = EF_model.predict_all()

    # Get evaluation
    mse = EF_model.evaluate(test_data)
    return predictions, mse


def MF_make_recommendation(ratings, prediction_matrix, user_id, k):
    """
    Generate recommendation from a dense user-item matrix
    :param ratings: user-item matrix
    :param prediction_matrix: predicted user-item matrix
    :param user_id: Index of the target user
    :param k: number of recommendation
    :return: id of the recommended items
    """
    # Retrieve user predictions
    user_ratings = ratings[user_id]
    items_unknown = np.argwhere(user_ratings == 0)
    user_prediction = prediction_matrix[user_id, items_unknown]

    # Retrieve and sort the items recommended (indexes of the columns)
    recommended_items = np.argsort(user_prediction, axis=0)[::-1]

    # Make recommendation: select the top k items
    recommendations = recommended_items[:k]
    return recommendations


if __name__ == "__main__":
    # Import dataset
    df = utils.import_dataset("./data")

    # Split the dataset into training and test sets, according to the "time" attribute
    train_df, test_df, common_users = utils.split_train_test(df, num_test_days=30)

    # Create the User-Item matrix
    train_ratings = utils.Dataframe2UserItemMatrix(train_df)
    test_ratings = utils.Dataframe2UserItemMatrix(test_df)

    # METHOD 1: Item-based Collaborative Filtering
    # Latent factors: Explicit Matrix Factorization
    print("\nRecommendation based on the CF method (Matrix Factorization) ...\nTraining...")
    train_predictions, test_mse = collaborative_filtering(train_ratings, test_ratings)
    print(f"\nPREDICTIONS {train_predictions.shape} generated with MSE (test data): {test_mse}")

    # TODO for all common users:
    # a. Make recommendations
    # b. Check if recommendations are equal to the target in the test sets
    # c) Compute the recall metric
    # TODO compute the overall recall

    # b) Make recommendation: higher predicted values
    index_user = 200
    num_recommendation = 10
    recommendations = MF_make_recommendation(train_ratings, train_predictions, index_user, num_recommendation)
    print(f"The {num_recommendation} items recommended for the user {index_user}\n", *recommendations)
