# Load libraries
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

features_train = read_csv('./../../../datasets/kannada_letters/train.csv')

# print(features_train.shape,features_validation.shape)
# print(features_train.head(20))
# set_option('precision',3)
# print(features_train.describe())
features_train = features_train.values
Y_train = features_train[:,0].astype(float)
X_train = features_train[:,1:785].astype(float)
print(features_train.shape)
import pickle

num_folds = 10
seed = 7
scoring = 'accuracy'
scaler = StandardScaler().fit(X_train)
pickle.dump(scaler,open("./../final_model_pkl_files/scaler.pkl","wb"))
rescaledX = scaler.transform(X_train)
pca = PCA(.75).fit(rescaledX)

pickle.dump(pca,open("./../final_model_pkl_files/pca.pkl","wb"))
rescaledX = pca.transform(rescaledX)


# t = time.time()
# model = KNeighborsClassifier(n_neighbors = 5)
# model.fit(rescaledX,Y_train)
# print(time.time()-t)


# pickle.dump(model,open("./../final_model_pkl_files/knn_kannada_digits_new.pkl","wb"))
t = time.time()
model = LogisticRegression(C = 1)
model.fit(rescaledX,Y_train)
print(time.time()-t)
pickle.dump(model,open("./../final_model_pkl_files/lr_kannada_digits.pkl","wb"))

# t = time.time()
# model = GradientBoostingClassifier(max_depth = 5,min_samples_split = 400)
# model.fit(X_train,Y_train)
# print(time.time()-t)
# pickle.dump(model,open("./../final_model_pkl_files/gbc_kannada_digits.pkl","wb"))
