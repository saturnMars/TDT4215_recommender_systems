import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from ExplicitMF import ExplicitMF as ExplicitMatrixFactorization
import numpy as np

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def predict_topk(ratings, similarity, kind='user', k=50):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][tuple(top_k_items)].dot(ratings[i, :][tuple(top_k_items)].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][tuple(top_k_items)]))

    return pred

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
    print("EXAMPLE of an event:\n", df.loc[1000, :])

    # Create the User-Item matrix
    ratings = utils.Dataframe2UserItemMatrix(df)

    # print("Numpy 2D-array is: ", ratings)

    # Split the dataset into training and test sets
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=99)

    # METHOD 1:
    # Model-based collaborative filtering (Latent factors: Explicit Matrix Factorization)
    # a) Train the model
    print("\nRecommendation based on the CF method (Matrix Factorization) ...\nTraining...")
    train_predictions, test_mse = collaborative_filtering(train_ratings, test_ratings)
    print(f"\nPREDICTIONS {train_predictions.shape} generated with MSE (test data): {test_mse}")

    # b) Make recommendation: higher predicted values
    index_user = 200
    num_recommendation = 10
    recommendations = MF_make_recommendation(train_ratings, train_predictions, index_user, num_recommendation)
    print(f"The {num_recommendation} items recommended for the user {index_user}\n", *recommendations)

    # METHOD 2:
    # Item-based CF and training the model
    print("\nRecommendation based on item based CF ...\n")
    # recommendation_item = itemCF_make_recommendation(ratings, train_predictions, index_user,num_recommendation)
    # train and test split is done.

    item_similarity = fast_similarity(train_ratings, kind='item')

    # print(item_similarity[:4, :4])
    pred = predict_topk(train_ratings, item_similarity, kind='item', k=40)

    print('Top-k Item-based CF MSE: ' + str(get_mse(pred, test_ratings)))

    #should return the top item instead of MSE, just basically followed the guide since I couldnt make it work with a similar
    #code as Håvard.

    #print("\nRecommended item(s): ")
    #print(recommend_items(ratings, 0, 5))