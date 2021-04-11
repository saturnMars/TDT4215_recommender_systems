import utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse as sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from pandas import read_json, concat

def content_based_single_item(documentId, dataset):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0)
    df = utils.import_dataset(dataset)
    df = df.drop_duplicates(subset=['documentId'])
    df = df[['documentId', 'words', 'title']]
    df_new = df.reset_index(drop=True)
    df_new['words'] = df_new['words'].apply(' '.join)
    #Could use title instead of words if it yields better performance
    tfidf_matrix = tf.fit_transform(df_new["words"])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    csm = cosine_similarities

    indices = pd.Series(df_new.index, index=df_new.documentId)

    def recommend(post, csm = csm):
        idx = indices[post]
        score_series = list(enumerate(csm[idx]))
        score_series = sorted(score_series, key=lambda x: x[1], reverse=True)
        # Value in [0:5] could be changed to get more reccomandations to increase recall
        score_series = score_series[0:5]
        post_indices = [i[0] for i in score_series]
        return post_indices
    

    recommended = recommend(documentId)
    
    recommended_items = list()

    for item in recommended:
        recommended_items.append(df_new.iloc[item]['documentId'])
    
    return recommended_items

def get_n_highest(file_path, column, n):
    daily_events = list()

    with open(file_path) as file:
            events = read_json(file, lines=True)
            daily_events.append(events)
    
    df = concat(daily_events, ignore_index=True)
    df.drop_duplicates(subset=['documentId'])
    
    most_recent = df[column].nlargest(n, keep = "all")
    index = most_recent.index
    documentIds = list()
    for i in index:
        documentIds.append(df.iloc[i]['documentId'])

    return documentIds
    
def content_based(user, dataset):
    path = os.path.join(dataset, user)
    #number of items (3) could be tweaked to increase performance
    #could use activeTime instead of time
    best_items = get_n_highest(path, 'time', 3)
    recommended_items = list()
    for item in best_items:
        recommended_items.extend(content_based_single_item(item, dataset))
    
    recommended_items = list(dict.fromkeys(recommended_items))

    return recommended_items

def mse(recommandations, user):
    daily_events = list()

    with open(user) as file:
            events = read_json(file, lines=True)
            daily_events.append(events)
    
    df = concat(daily_events, ignore_index=True)
    df.drop_duplicates(subset=['documentId'])

    mistakes = 0

    user_items = df['documentId'].tolist()

    for item in recommandations:

        mistakes += 1

        if item in user_items:
            mistakes -= 1
    
    mse = mistakes/len(recommandations)

    return mse



def recall(recommandations, user):

    daily_events = list()

    with open(user) as file:
            events = read_json(file, lines=True)
            daily_events.append(events)
    
    df = concat(daily_events, ignore_index=True)
    df.drop_duplicates(subset=['documentId'])

    correct_predictions = 0

    user_items = df['documentId'].tolist()

    for item in recommandations:

        if item in user_items:
            correct_predictions += 1
    
    recall = correct_predictions/len(user_items)

    return recall

def ctr(recommandations, user):
    daily_events = list()

    with open(user) as file:
            events = read_json(file, lines=True)
            daily_events.append(events)
    
    df = concat(daily_events, ignore_index=True)
    df.drop_duplicates(subset=['documentId'])

    correct_predictions = 0

    user_items = df['documentId'].tolist()

    for item in recommandations:

        if item in user_items:
            correct_predictions += 1
    
    ctr = correct_predictions/len(recommandations)

    return ctr


if __name__ == "__main__":

    mse_avg = 0
    recall_avg = 0
    ctr_avg = 0

    #how many users to get reccomandations on. The higher number, the more accurate performance metrics
    number_of_users = 2

    #needs to be a directory containing one file for each user.
    dataset = "data/pre/train"

    for filename in os.listdir(dataset)[:number_of_users]:
        print(filename)
    
        recommandations = content_based(filename, dataset)
    
        mse_value = mse(recommandations, os.path.join(dataset, filename))
        recall_value = recall(recommandations, os.path.join(dataset, filename))
        ctr_value = ctr(recommandations, os.path.join(dataset, filename))
    
        print(mse_value)
        print(recall_value)
        print(ctr_value)

        mse_avg += mse_value
        recall_avg += recall_value
        ctr_avg += ctr_value

    mse_avg = mse_avg/number_of_users
    recall_avg = recall_avg/number_of_users
    ctr_avg = ctr_avg/number_of_users
    f1_score = 2*(recall_avg*ctr_avg)/(recall_avg+ctr_avg)


    print("\n Performance: \n")
    print("MSE: ")
    print(mse_avg)
    print("\n")
    print("Recall: ")
    print(recall_avg)
    print("\n")
    print("CTR: ")
    print(ctr_avg)
    print("\n")
    print("F1 score: ")
    print(f1_score)
    
    