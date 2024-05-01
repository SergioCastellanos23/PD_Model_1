# SERGIO ANTONIO CASTELLANOS TORRES
#### Consultor de riesgo - Ciencia de Datos | Credit Risk Analysis| Agribusiness

Este portafolio es creado con fines profesionales, medir mi capacidad como Consultor de riesgo y ciencia de datos. 
Estoy entusiasmado de aportar a este mundo del data science y al igual me encantaría tener retroalimentación de profesionales en el tema.

- Mail: sergio.castellanost23@gmail.com
- LinkedIn Profile: www.linkedin.com/in/sergio-antonio-castellanos-torres-828946126


## EDUCACIÓN 
- Agronegocios Internacionales | Universidad Veracruzana | 2021

## EXPERIENCIA 
 
### Consultor de riesgo y Ciencia de Datos 
- 3PI, LLC · ene. 2023 - actualidad ·

Desempeño el rol como Consultor de Riesgo y Ciencia de datos dentro de 3PI, LLC, en el área de Services (Project & Program Management) dentro de todas las fases de desarrollo de modelos.

- Apoyo al equipo de modelado en la limpieza, transformación y visualización de datos con Pandas, Numpy, Matplotlib y Seaborn en la detección de valores atípicos, imputación de datos y transformación o agrupación de variables categóricas para la construcción de modelos.
- Proporcionar insights con análisis de estadística descriptiva e inferencial en el proceso de selección de variables con un enfoque hacia la construcción de un modelo parsimonioso y monotónico (Information Value, Correlaciones, Chi Squared Test, Variable Clustering) con OptimalBinning, Matplotlib, Statsmodels, VarClusHi.
- Construcción de modelos de Credit Score (Logistic Regression, Decision Trees, Random Forest Classifier, XGB Classifier, AdaBoosClassifier) para incrementar la aprobación de solicitudes en el sector de crédito, manteniendo el riesgo de incumplimiento y aumentando la rentabilidad.
- Consultar y capturar datos utilizando bases de datos como PostgreSQL para acceder a la información necesaria en mis proyectos.


### CURSOS
- Intermediate Machine Learning | Kaggle
- Python
- PostgreSQL
- Excel
- PowerBI (Cursando)
- PySpark (Proximamente)


## PREDICT LIKELIHOOD

- El proyecto denominado "Predict Likelihood" fue un proyecto con datos limitados:
  
- Por ello, derivamos a tomar medidas muy creativas. Por ende, se tomó la decisión de evaluar el rendimiento de las variables en cada modelo.
- Utilizamos Chi Square Test, Correlación (Spearman), Information Value, Random Forest, Decision Trees, XGBoost, entre otros.
- Para la detección de outliers utilizamos el rango intercuantil, sin embargo, ampliamos el umbral a 3.5 veces (valores extremos). A su vez Histogramas y graficos de cajas para analizar la distribución de los datos y considerar en si eliminar los valores atipicos detectados anteriormente.
- En conjunto analizamos los graficos de disperción con linea de regresión podemos observar la tendencia hacia DEFAULT, por ejemplo, DURATION, mientras mayor sea el termino, mayor la probabilidad de incumplimiento.
- En la variable AGE decidimos mantener solo la población menor a 67. Considerando a la población mayor a 67 como de "alto riesgo".
- Medimos el rendimiento de las variables en cada proceso (Chi Square, Correlación, Variable Clustering, Information Value), Forward, Backward y Stepwise Selection. Y posteriormente utilizando GridSearchCV para encontrar los mejores hiperparametros por cada modelo (Logistic Regression, Random Forest Classifier, XGB Classifier, Gradient Boosting Classifier, Decision Tree Classifier, AdaBoost Classifier y aprovechar al máximo el proceso de selección de variables, además Lasso, Ridge y ElasticNet.
- Posteriormente, la selección del modelo se dividió la muestra en Train y Test (70/30), utilizando los hiperparametros del paso de Variable Selection. Seleccionamos Logistic Regression.
- Transformamos los datos en ciertas variables (analizando los datos proporcionados por WOE y recrear monoticidad en los datos). 
- Al ejecutar el rendimiento de las variables en los modelos de ML se logró apreciar cuales variables son fuertes predictores y su impacto constante en cada modelo, dichas variables fueron CHK_ACCT, DURATION. Los  demás predictores se evaluaron con su rendimiento en los diferentes modelos y los diferentes Test como fue Information Value,  Chi Square, Correlación.
- El modelo se ejecutó mediante Regresión Logística, por su rendimiento en los 5 CV, un mean square error de los más bajos dentro de la selección de modelos. Con un accuracy mayor a 75%, aceptable para la cantidad limitada de datos.
- Con los coeficientes se utilizó la formula de regresión logistica, multiplicando el coeficiente por el valor otorgado. Las variables que mayor repercuten en son CHK_ACCT, DURATION, USED_CAR y GUARANTOR.
- Se creo un Scorecard para fines de otorgamiento de crédito a nuevos clientes con las mejores variables.
- Como validación del modelo se utilizo Gini, ROC Curve y KS, dando como resultado un 0.77 ROC AUC, KS= 48.59% y Gini de 0.55, considerandose aceptable en la predicción.

## Credit Cars

- Un dataset con más de 65,000 aplicantes para evaluar la probabilidad de morosidad e igual, aplicar un Scorecard:
 
- Notifiqué algunas tendencias que fueron de alto impacto y que se deberían tomar en cuenta.
- Demasiados datos nulos y por ende, tomar la decisión de imputar algunos datos.
- Transformamos variables por categorias, por ejemplo, 'STATE' fue dividida por zonas 'NORTH_STATES', 'MIDWEST_STATES', 'SOUTH_STATES', 'WEST_STATES' para simplificar nuestros datos, ya que al realizar nuestros analisis sería complicado analizar su importancia.
- 
