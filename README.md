# Kannada Digits Classification

Used Standard Scaler to standardize data and PCA to reduce the number of features.
Information retained in PCA is 75%.
Used KNN algorithm with n_neighbors = 5 the training time was around 35s and results i got for 10000 testing examples :

           accuracy = 0.9794
                precision    recall  f1-score   support
           0       0.95      0.93      0.94      1000
           1       0.93      1.00      0.96      1000
           2       1.00      0.97      0.99      1000
           3       0.99      0.97      0.98      1000
           4       0.98      1.00      0.99      1000
           5       1.00      0.98      0.99      1000
           6       0.98      0.99      0.98      1000
           7       0.98      0.98      0.98      1000
           8       1.00      0.99      0.99      1000
           9       1.00      0.98      0.99      1000

           micro avg       0.98      0.98      0.98     10000
           macro avg       0.98      0.98      0.98     10000
           weighted avg       0.98      0.98      0.98     10000

          CONFUSION MATRIX
          [[929  68   0   1   0   0   0   1   1   0]
           [  1 999   0   0   0   0   0   0   0   0]
           [ 17   2 975   3   1   0   1   1   0   0]
           [ 18   2   0 971   3   1   0   5   0   0]
           [  1   0   0   0 997   1   0   1   0   0]
           [  0   0   1   2  11 985   0   1   0   0]
           [  0   0   0   1   1   0 987  10   0   1]
           [  2   3   0   4   0   0  13 978   0   0]
           [  8   3   0   0   1   0   0   0 988   0]
           [  0   0   0   1   1   0  10   2   1 985]]



.There was potential in LogisticRegression and GradientBoostingClassifier
but couldn't train on 60000 training examples on my PC i thik GradientBosstingClassifier would've given better results.
