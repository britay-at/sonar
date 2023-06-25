import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer

import warnings

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/Datos M.csv')
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    # Eliminar las muestras con valores faltantes
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test[X_test.index]

    # Resultados con los datos originales
    estimators = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN': KNeighborsClassifier(),
        'DecisionTreeClf': DecisionTreeClassifier(),
        'RandomTreeForest': RandomForestClassifier(random_state=0)
    }

    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_class.predict(X_test)
        print('=' * 64)
        print('Datos Originales')
        print('SCORE Bagging with {}: {}'.format(name, accuracy_score(bag_predict, y_test)))

    # Resultados con los datos normalizados y discretizados
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    boosting_normalized = GradientBoostingClassifier(n_estimators=188).fit(X_train_normalized, y_train)
    boosting_pred_normalized = boosting_normalized.predict(X_test_normalized)
    print('=' * 64)
    print('Datos Normalizados')
    print(accuracy_score(boosting_pred_normalized, y_test))

    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_normalized)
    X_test_discretized = discretizer.transform(X_test_normalized)

    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train_discretized, y_train)
        bag_predict = bag_class.predict(X_test_discretized)
        print('=' * 64)
        print(' Discretizados')
        print('SCORE Bagging with {}: {}'.format(name, accuracy_score(bag_predict, y_test)))

