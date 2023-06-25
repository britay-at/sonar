# Importamos las bibliotecas
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # Importamos el dataset del 2017
    dataset = pd.read_csv('./data/Datos M.csv')

    # Mostramos el reporte estadístico
    #print(dataset.describe())

    # Convertir la columna 'Toxicos' a tipo numérico
    dataset['%Toxicos'] = pd.to_numeric(dataset['%Toxicos'], errors='coerce')

    # Verificar los tipos de datos después de la conversión
    #print(dataset.dtypes)

    # Seleccionamos los features que vamos a usar
    features = ['Total commits', 'Total commits per day', 'Accumulated commits', 'Committers', 'Committers Weight', 'suma',
                'classes', 'ncloc', 'functions', 'duplicated_lines', 'test_errors', 'skipped_tests', 'coverage', 'complexity',
                'comment_lines', 'comment_lines_density', 'duplicated_lines_density', 'files', 'directories', 'file_complexity',
                'violations', 'duplicated_blocks', 'duplicated_files', 'lines', 'public_api', 'statements', 'blocker_violations',
                'critical_violations', 'major_violations', 'minor_violations', 'info_violations', 'lines_to_cover', 'line_coverage',
                'conditions_to_cover', 'branch_coverage', 'sqale_index', 'sqale_rating', 'false_positive_issues', 'open_issues',
                'reopened_issues', 'confirmed_issues', 'sqale_debt_ratio', 'new_sqale_debt_ratio', 'code_smells', 'new_code_smells',
                'bugs', 'effort_to_reach_maintainability_rating_a', 'reliability_remediation_effort', 'reliability_rating',
                'security_remediation_effort', 'security_rating', 'cognitive_complexity', 'new_development_cost', 'security_hotspots',
                'security_review_rating']

    X = dataset[features]
    y = dataset['%Toxicos']

    # Manejo de valores faltantes
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X = imputer.fit_transform(X)
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)

   # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

 
    # Creamos y ajustamos el modelo de regresión lineal
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Creamos y ajustamos el modelo de Lasso
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    # Creamos y ajustamos el modelo de Ridge
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    # Creamos y ajustamos el modelo de ElasticNet
    modelElasticNet = ElasticNet(random_state=0, max_iter=2000, alpha=0.5).fit(X_train, y_train)
    y_pred_elastic = modelElasticNet.predict(X_test)

    # Calculamos el error medio cuadrado para cada modelo
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)
    # Mostramos las pérdidas de cada modelo
    print("Linear Loss:", linear_loss)
    print("Lasso Loss:", lasso_loss)
    print("Ridge Loss:", ridge_loss)
    print("ElasticNet Loss:", elastic_loss)

    # Mostramos los coeficientes de cada modelo
    print("="*32)
    print("Coeficientes linear:")
    print(modelLinear.coef_)

    print("="*32)
    print("Coeficientes lasso:")
    print(modelLasso.coef_)

    print("="*32)
    print("Coeficientes ridge:")
    print(modelRidge.coef_)

    print("="*32)
    print("Coeficientes elastic net:")
    print(modelElasticNet.coef_)

    # Calculamos la exactitud (score) de cada modelo
    print("="*32)
    print("Score Lineal:", modelLinear.score(X_test, y_test))
    print("Score Lasso:", modelLasso.score(X_test, y_test))
    print("Score Ridge:", modelRidge.score(X_test, y_test))
    print("Score ElasticNet:", modelElasticNet.score(X_test, y_test))
    #la parte economica da mayor peso el ridge
    #elastic net todas las variables se anulan 
    #me da valor negativo por que es el valor bajo de las variable 
