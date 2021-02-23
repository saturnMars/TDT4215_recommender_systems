# TDT4215_recommender_systems
**TASK**: Develop a news recommender system

**DATA**: user data from the local digital newspaper (January - March)
- 1,000 most active users 
- 9 attributes (userID, title) 

**Examples**:
a. CF --> Explicit Matrix Factorization (MF)
b. Content-based --> TF-IDF + Cosine similarity 

**EVALUATION**:
1. Recall (hit rate) --> positive instances that are correctly predicted 
2. CTR (Click- through rate) --> recommendations clicked over total rec
3. ARHR (average reciprocal hit rate) 
4. MSE 

**REPORT** (max 15 pages.)
a. Motivation
b. overview of existing news recommender algorithms and approaches
c. evaluation results
d. conclusion 

**EVALUATION**: report + presentation + performance evaluation

**STEPS**
1.train test split --> scikit-learn
2. Implement a Collaborative filtering RS
3. Implement a Content-based RS
4. Merge them and build a Hybrid RS
5. Evaluation Recall, ARHR)
