import pickle
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

model = pickle.load(open("./../final_model_pkl_files/lr_kannada_digits.pkl","rb"))

dataset = read_csv("./../../../datasets/kannada_letters/train.csv")
X_validation = dataset.values[:,1:785].astype(float)
Y_validation = dataset.values[:,0].astype(float)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score

num_folds = 10
seed = 7
scoring = 'accuracy'
scaler = pickle.load(open("./../final_model_pkl_files/scaler.pkl","rb"))
rescaledX = scaler.transform(X_validation)
pca = pickle.load(open("./../final_model_pkl_files/pca.pkl","rb"))
rescaledX = pca.transform(rescaledX)

pred = model.predict(rescaledX)
Y_validation = Y_validation
print("accuracy",accuracy_score(Y_validation,pred))
print(classification_report(Y_validation,pred))
print(confusion_matrix(Y_validation,pred))

