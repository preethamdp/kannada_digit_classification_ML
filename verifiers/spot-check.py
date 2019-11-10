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
features_validation = read_csv('./../../../datasets/kannada_letters/train.csv')

# print(features_train.shape,features_validation.shape)
# print(features_train.head(20))
# set_option('precision',3)
# print(features_train.describe())
features_train = features_train.values
Y_train = features_train[:,0].astype(float)
X_train = features_train[:,1:785].astype(float)
print(features_train.shape)
results = []
names = []
pipelines = []

pipelines.append(('RF75',Pipeline([('Scaler',StandardScaler()),('PCA',PCA(.75)),('LR',RandomForestClassifier())])))
pipelines.append(('ADA75',Pipeline([('Scaler',StandardScaler()),('PCA',PCA(.75)),('knn',AdaBoostClassifier())])))

pipelines.append(('GD75',Pipeline([('Scaler',StandardScaler()),('PCA',PCA(.75)),('LR',GradientBoostingClassifier())])))
pipelines.append(('ET75',Pipeline([('Scaler',StandardScaler()),('PCA',PCA(.75)),('knn',ExtraTreesClassifier())])))

num_folds = 10
seed = 7
scoring = 'accuracy'
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train[:2000], Y_train[:2000], cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    if cv_results.mean()>0.8:
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

fig = pyplot.figure()
config_name = 'ensembles comparision'
fig.suptitle(config_name)
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.savefig(config_name+".png")
pyplot.show()