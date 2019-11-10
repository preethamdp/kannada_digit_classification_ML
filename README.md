# Kannada Digits Classification

Used Standard Scaler to standardize data and PCA to reduce the number of features.
Information retained in PCA is 75%.
Used LogisticRegression algorithm with C = 1 the training time was around 335s and results i got for 60k testing examples :

          accuracy 0.9668833333333333
              precision    recall  f1-score   support

         0.0       0.97      0.96      0.96      6000
         1.0       0.97      0.98      0.98      6000
         2.0       1.00      0.99      0.99      6000
         3.0       0.94      0.94      0.94      6000
         4.0       0.96      0.98      0.97      6000
         5.0       0.98      0.97      0.98      6000
         6.0       0.94      0.96      0.95      6000
         7.0       0.94      0.92      0.93      6000
         8.0       0.99      0.98      0.99      6000
         9.0       0.99      0.97      0.98      6000

         micro avg       0.97      0.97      0.97     60000
         macro avg       0.97      0.97      0.97     60000
         weighted avg       0.97      0.97      0.97     60000

           [[5746  160    2   48    8    4    4    5   20    3]
            [  30 5909    0   19    6   20    2    3    1   10]
            [  24    2 5957   11    0    4    0    1    1    0]
            [  72    4    8 5656   57   42   31  116    5    9]
            [   4    2    0   32 5863   59    6   13    8   13]
            [   0   11    7   41   67 5850    9    3   12    0]
            [   2    1    0   44    5    3 5756  173    1   15]
            [  18    8    0  146   22    8  258 5533    3    4]
            [  36    7    1    9   15    6    6    0 5902   18]
            [   3    0    0    3   45    2   58   32   16 5841]]



There was also potential in many ensembles
