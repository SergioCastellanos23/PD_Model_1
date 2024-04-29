# SERGIO ANTONIO CASTELLANOS TORRES

Este portafolio es creado con fines profesionales, medir mi capacidad como Consultor de riesgo y ciencia de datos. 
Estoy entusiasmado de aportar a este mundo del data science y al igual me encantaría tener retroalimentación de profesionales en el tema.

- Mail: sergio.castellanost23@gmail.com
- LinkedIn Profile: www.linkedin.com/in/sergio-antonio-castellanos-torres-828946126

### Consultor de riesgo - Ciencia de Datos | Credit Risk Analysis| Agribusiness

## EDUCACIÓN 
- Agricultural Business | Universidad Veracruzana | 2021

## EXPERIENCIA 
 
Consultor de riesgo y Ciencia de Datos 
- 3PI, LLC · ene. 2023 - actualidad ·

Desempeño el rol como Consultor de Riesgo y Ciencia de Datos dentro de 3PI, LLC, en el área de Services (Project & Program Management). Una de mis funciones principales es la creación de modelos estadísticos y predictivos parsimoniosos para el sector de riesgo crediticio incluyendo Scorecards, dicho análisis se diseña con limitados o grandes volúmenes de datos dependiendo del rendimiento de las variables en el modelado buscando equilibrio y predicción, con ello mejorando la interpretabilidad.

- Limpieza y transformación de los datos.
- Detección de Outliers, Impute data, Categorical variables.
- Análisis de estadística descriptiva e inferencial para la calidad e importancia de las variables para su selección y modelado con Matplotlib, Statsmodels, Sckit-Learn, Pengouin. 
- Colaboración en la creación de modelos.
- Desarrollo de Modelos Tradicionales y Modelos Machine Learning (Modelos parsimoniosos).
- Chi Square Test, Correlaciones, Regresión Logística, Regresión Lineal, Information Value, Variable Clustering, Decision Trees, Random Forest, XGBoost, Lasso, ElasticNet, Ridge, Forward, Backward, Stepwise.
- Estimación de Probability of Default y actualmente con Early Probability of Default. 
- Creación de Scorecards generando una precisión de modelo aceptable y reduciendo significativamente el riesgo.
- Comunicar insights, hallazgos y conclusiones derivadas del proceso (Data Storytelling).
- Consultas en PostgreSQL

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
