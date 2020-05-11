import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from datetime import datetime
import xgboost as xgb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

# from sklearn.metrics import confusion_matrix, mean_squared_error


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


df = pd.read_csv('/Users/ast/Desktop/labelled_1.csv')
df.drop(df.columns[[14, 15]], axis=1, inplace = True)
labels = df[df.columns[-1]]

labels = labels.map({ 'BENIGN' : 0, 'SSH-Patator' : 1, 'FTP-Patator' : 1})
print(labels.describe())
print(labels.value_counts())


features = df[df.columns[:-1]]
#
# # features.info()
#
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)
xgb = xgb.XGBClassifier(learning_rate=0.1, n_estimators=80, max_depth=4,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

# params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#         }
# folds = 3
# param_comb = 5
# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,Y_train), verbose=3, random_state=27)
# start_time = timer(None) # timing starts from this point for "start_time" variable
# search = random_search.fit(X_train, Y_train)
# timer(start_time)
# print(search.best_params_)


print('Start training XGB')
start_time = timer(None)
xgb.fit(X_train, Y_train, eval_metric='auc')
timer(start_time)
print("Start predicting XGB")
start_time = timer(None)
predictions = xgb.predict(X_test)
timer(start_time)
pred_proba = xgb.predict_proba(X_test)[:, 1]
print('Statistics')
print("AUC : %f" % metrics.roc_auc_score(Y_test, pred_proba))
print("F1 Score: %f" % metrics.f1_score(Y_test, predictions))
print('**********************************')

clf = LogisticRegression(C=1e5)
print('Start training log regression')
start_time = timer(None)
clf.fit(X_train, Y_train)
timer(start_time)
print("Score: ", clf.score(X_test, Y_test))
start_time = timer(None)
print("Start predicting log regression")
y_pred = clf.predict(X_test)
timer(start_time)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
print('Statistics')
print("AUC Score is: {}".format(metrics.roc_auc_score(Y_test, y_pred_proba)))
print("F1 score is: {}".format(metrics.f1_score(Y_test, y_pred)))
print('**********************************')


# parameters = {'n_estimators': [10, 20, 30, 50], 'max_depth': [2, 3, 4]}
clf = RandomForestClassifier(max_depth=4, n_estimators=20)
# clf = GridSearchCV(alg, parameters, n_jobs=4)
print('Start training Random Forest')
start_time = timer(None)
clf.fit(X_train, Y_train)
timer(start_time)
print("Score: ", clf.score(X_test, Y_test))
start_time = timer(None)
print("Start predicting Random Forest")
y_pred = clf.predict(X_test)
timer(start_time)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
print('Statistics')
print("AUC Score is: {}".format(metrics.roc_auc_score(Y_test, y_pred_proba)))
print("F1 score is: {}".format(metrics.f1_score(Y_test, y_pred)))
print('**********************************')


clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100,))
print('Start training simple NN')
start_time = timer(None)
clf.fit(X_train, Y_train)
timer(start_time)
print("Score: ", clf.score(X_test, Y_test))
print('Start predicting simple NN')
start_time = timer(None)
y_pred = clf.predict(X_test)
timer(start_time)
print('Statistics')
y_pred_proba = clf.predict_proba(X_test)[:, 1]
print("AUC Score is: {}".format(metrics.roc_auc_score(Y_test, y_pred_proba)))
print("F1 score is: {}".format(metrics.f1_score(Y_test, y_pred)))
print('**********************************')


clf = SVC(kernel='linear')
print('Start training SVM')
start_time = timer(None)
clf.fit(X_train, Y_train)
timer(start_time)
print("Score: ", clf.score(X_test, Y_test))
print('Start predicting SVM')
start_time = timer(None)
y_pred = clf.predict(X_test)
timer(start_time)
print('Statistics')
y_pred_proba = clf.predict_proba(X_test)[:, 1]
print("AUC Score is: {}".format(metrics.roc_auc_score(Y_test, y_pred_proba)))
print("F1 score is: {}".format(metrics.f1_score(Y_test, y_pred)))
print('**********************************')

# добавить сохранение моделей?
# тип посчитали, выбрали лучшую, дальше смотрим как быстрее считать её с FL или нет