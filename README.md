# Proyectos

## Predict Likelihood

El proyecto "Predict Likelihood" tenía como objetivo predecir la probabilidad de incumplimiento utilizando un conjunto de datos limitado. Dadas las limitaciones, el proyecto empleó metodologías creativas y rigurosas para evaluar el rendimiento de las variables y optimizar la selección del modelo.

## Metodologías y Técnicas


### Detección y manejo de valores atípicos
- **Rango intercuartil (IQR)**: umbral ampliado a 3,5 veces para identificar valores atípicos extremos.
- **Histogramas y diagramas de caja**: visualización de la distribución de datos e identificación de posibles valores atípicos para su eliminación.

### Análisis de los datos
- **Gráficos de dispersión con líneas de regresión**: tendencias ilustradas, como la relación entre DURATION y DEFAULT.

### Selección de Variables
- **Test Chi-Cuadrado**: Se evaluó la independencia de variables.
- **Correlación de Spearman**: Se evaluaron las relaciones monótonas entre variables.
- **Valor de la Información (IV)**: Se determinó el poder predictivo de las variables.
- **Bosque aleatorio, árboles de decisión, XGBoost**: Se utiliza para comprender la importancia y la interacción de las variables.

### Creación de Scorecard
- Desarrollé un Scorecard para la evaluación crediticia de nuevos clientes utilizando las variables más predictivas.
