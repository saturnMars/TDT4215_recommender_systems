# TDT4215: Recommender Systems (Group 7)

## TASK: Develop a news recommender system

## DATA: user data from the local digital newspaper (January - March)

- 1,000 most active users
- 9 attributes (userID, title)

## Examples

- CF --> Explicit Matrix Factorization (MF)
- Content-based --> TF-IDF + Cosine similarity

## EVALUATION

1. Recall (hit rate) --> positive instances that are correctly predicted
2. CTR (Click- through rate) --> recommendations clicked over total rec
3. ARHR (average reciprocal hit rate)
4. MSE

## REPORT (max 15 pages)

1. Motivation
2. overview of existing news recommender algorithms and approaches
3. evaluation results
4. conclusion

## EVALUATION: report + presentation + performance evaluation

## STEPS

1. train test split --> scikit-learn
2. Implement a Collaborative filtering RS
3. Implement a Content-based RS
4. Merge them and build a Hybrid RS
5. Evaluation Recall, ARHR)

## TODO for Content-based Filtering

> Implement the article from
> https://medium.com/@bindhubalu/content-based-recommender-system-4db1b3de03e7
> or from
> https://towardsdatascience.com/content-based-recommender-systems-28a1dbd858f5

1. Stopword removal in titles
2. Merge titles and keyword lists
3. Only work with single user file:
4. TF\*IDF for keywords compared to the keywords of all their articles
5. Build matrix from the article

   ```asciiart
       doc0       doc1   doc2   doc3
   w0  tfidf00
   w1              ...
   w2                           tfidf23
   ```

6. Compute cosine similarity between items
7. Recommend from the article
