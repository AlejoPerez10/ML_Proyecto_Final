# 🧠 Proyecto de Modelo Supervisado Predictivo

## Contexto del Proyecto

- En este proyecto trabajé en un modelo supervisado para predecir si una persona podría tener enfermedad cardíaca o no, usando información clínica y datos personales de los pacientes. Para esto, usé el Heart Disease Dataset que saqué de Kaggle ```https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset```. Elegí esta base de datos porque ya había trabajado con ella en otro proyecto de programación y además me pareció muy interesante y única para este trabajo.

- Durante el desarrollo del proyecto, realicé varias etapas: primero cargué y limpié los datos, luego hice un análisis exploratorio para entenderlos mejor, después construí nuevas características con ingeniería de datos, entrené y evalué varios modelos supervisados, y finalmente desplegué el modelo mediante una API. Además, creé una app en Streamlit que permite ver los resultados del modelo de forma gráfica y fácil de usar.

- El objetivo de todo esto es poder tener un modelo que ayude a predecir el riesgo de enfermedad cardíaca en pacientes, para que profesionales de la salud o investigadores puedan tomar decisiones más informadas basadas en los datos.

## 📁 Estructura del Proyecto (completar)

```
ML_Proyecto_Final/
├── entorno_ml-venv/                         # Entorno virtual con todas las librerías necesarias.
├── mlops_pipeline/
│   └── src/
│       ├── __pycache__/                     # Archivos compilados de Python para rápida ejecución.
│       │
│       ├── API/                             # Contiene los archivos Docker para la API.
│       │   ├── Dockerfile.api
│       │   ├── Dockerfile.streamlit
│       │
│       ├── data/                            # Almacena los datasets y pruebas del modelo.
│       │   ├── resutls_history/             # historico de resultados guardados con fecha.
│       │   ├── best_model.pkl               # 
│       │   ├── feature_list.txt             # 
│       │   ├── model_comparison.png         # 
│       │   ├── model_metrics.csv            # 
│       │   ├── pipeline_preprocessor.pkl    # 
│       │   ├── results.csv                  # 
│       │   ├── test.csv                     # 
│       │   ├── train.csv                    # 
│       │
│       ├── Cargar_datos.ipynb               # Carga y preprocesamiento inicial de los datos.
│       ├── comprension_eda.ipynb            # Análisis exploratorio de datos (EDA).
│       ├── data_loader.py                   # Archivo para cargar una sola vez el df.
│       ├── ft_engineering.py                # Ingeniería de características.
│       ├── heuristic_model.py               # Modelo base o heurístico para comparación.
│       ├── model_deploy.py                  # Despliegue del modelo.
│       ├── model_evaluation.ipynb           # Evaluación del modelo.
│       ├── model_monitoring.py              # Entrenamiento del modelo.
│       ├── model_training_evaluation.py     # Monitoreo del modelo en producción.
│       ├── streamlit_app.py                 # Interfaz gráfica de mi app.
│
├── .dockerignore                            # Archivos a ignorar dentro de Docker.
├── .gitignore                               # Archivos a ignorar dentro de Git.
├── Base_de_datos.csv                        # Fuente principal de datos.
├── config.json                              # Configuraciones globales del proyecto.
├── readme.md                                # Este archivo.
├── requirements.txt                         # Librerías necesarias.
├── set_up.bat                               # Script para entorno de ejecución en Windows.
```

## Descripción general del Dataset

- El Heart Disease Dataset proviene de estudios médicos realizados en 1988 en cuatro lugares (Cleveland, Hungría, Suiza y Long Beach). Contiene información clínica de pacientes, con 14 atributos más relevantes (como edad, sexo, presión arterial, colesterol, frecuencia cardíaca, entre otros).
El objetivo del conjunto de datos es predecir la presencia de una enfermedad cardíaca, indicada por la variable “target” (0 = sin enfermedad, 1 = con enfermedad).

### Columnas de mi Dataset

**1-** age `Edad`

**2-** sex `Sexo`

**3-** chest pain type (4 values) `Tipo de dolor en el pecho`
- 0 - Angina Típica
- 1 - Angina Atípica
- 2 - Dolor no angionoso
- 3 - Asintómatico

**4-** resting blood pressure
   `Presión arterial en reposo`

**5-** serum cholestoral in mg/dl
   `Colesteron sérico`

**6-** fasting blood sugar > 120 mg/dl
   `Azúcar en ayunas`

**7-** resting electrocardiographic results (values 0,1,2) `Resultados del electrocardiograma en reposo`
- 0 - Normal
- 1 - Anomalía de la onda ST-T
- 2 - Hipertrofia ventricular izquierda

**8-** maximum heart rate achieved
   `Frecuencia cardiaca máxima aclanzada`

**9-** exercise induced angina
   `Angina inducida por ejercicio`

**10-** oldpeak = ST depression induced by exercise relative to rest
   `Depresión del segmenteo ST provocado por ejercicio`

**11-** the slope of the peak exercise ST segment
   `Pendiente del segmento ST`
- 0 - ascendente
- 1 - plano
- 2 - descendente

**12-** number of major vessels (0-3) colored by flourosopy
   `Número de vasos principales (observados por fluoroscopia)`

**13-** thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
   `Tipo de talsemio o defecto sanguíneo`
- 0 - normal
- 1 - defecto fijo (permanente)
- 2 - defecto reversible (mejora con esfuerzo o tratamiento)

**14-** Target `Objetivo`
- 0 - (NO) Enfermedad cardiaca ausente
- 1 - (SÍ) Enfermedad cardiaca presente

### Categorías de las variables

- **`Numéricas →`** Son cantidades medibles. Se pueden sumar o promediar.
Ej: edad, presión, colesterol, frecuencia cardíaca, oldpeak → como medir peso o temperatura.

- **`Categóricas →`** Representan grupos o etiquetas, no cantidades.
Ej: sex (h/m), fbs (sí/no), exang (sí/no), target (enfermo/sano), ca (0–3 vasos).

- **`Categóricas nominales →`** Son categorías sin orden natural.
Ej: cp, restecg, thal → tipos distintos (no mejores ni peores entre sí).

- **`Categórica ordinal →`** Son categorías con orden lógico.
Ej: slope (0 ascendente, 1 plana, 2 descendente).

## 🐍 Entorno Virtual y Activación

- Para este proyecto creé mi propio entorno virtual con todas las dependencias y librerías necearias para no tener ningún problema a la hora de trabajar con él desde un pc remoto, para activarlo haz los siguientes pasos:

1️⃣ Abrir Powershell (terminal) y navegar a la raíz del proyecto.
   ```bash
   cd C:\Users\user\ML_Proyecto_Final
   ```

2️⃣ Ejecutar el setup
   ```bash
   .\set_up.bat
   ```

3️⃣ Ajustar permisos (si es necesario)
   -Solo una vez, si da error al activar el entorno virtual
   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4️⃣ Verificar
   - Activar el entorno:
   ```bash
   .\entorno_ml-venv\Scripts\Activate.ps1
   ```

   - Revisar librerías instaladas:
   ```bash
   pip list
   ```

   - Abrir Jupyter y seleccionar el kernel:
   **entorno_ml-venv Python ETL**

❗Desactivar el entorno
   ```bash
   deactivate
   ```

## Carga de Datos (Cargar_datos.ipynb) y (data_loader.py)
- En este archivo me encargué de cargar y revisar el dataset que usaré en el proyecto. Primero importé todas las librerías necesarias y luego usé mi función cargar_datos del archivo data_loader.py, que creé para no tener que cargar los datos repetidamente y poder usarla donde la necesite.

Verifiqué que los datos se cargaran correctamente observando el tamaño del dataframe y las primeras filas con df.shape y df.head(5). También revisé los tipos de datos y pude apreciar que en mi dataframe no habían valores nulos usando df.info() y df.isnull().sum().

Por último, exploré algunas estadísticas generales con df.describe(include="all").T y noté que:
*la mayoría de los pacientes tiene alrededor de 54 años, casi el 70 % son hombres, el colesterol promedio es alto (246 mg/dl) y la presión en reposo también elevada (131 mmHg). La frecuencia cardíaca máxima promedio es de 149 bpm y cerca de la mitad de los pacientes presenta riesgo de enfermedad cardíaca (target ≈ 0.5). En general, esto refleja una población adulta con varios indicadores de riesgo cardiovascular.*

## Análisis Exploratorio de Datos (comprension_eda.ipynb)
**1. Exploración inicial**

Primero revisé los datos y clasifiqué las variables: numéricas (age, trestbps, chol, thalach, oldpeak), categóricas (sex, fbs, exang, target, ca), categóricas nominales (cp, restecg, thal) y ordinal (slope).

*No encontré valores nulos y no había columnas irrelevantes, así que los datos estaban listos para el análisis. Ajusté los tipos de datos correctamente: numéricas a int/float, categóricas como category y ordinal con su orden lógico.*

**2. Análisis univariable**

Revisé estadísticas descriptivas y visualicé histogramas, KDE y boxplots:

- Edad: distribución centrada en 50–60 años.

- Presión y colesterol: algunos outliers, colesterol sesgado a la derecha.

- Frecuencia cardíaca máxima (thalach) y oldpeak: thalach casi normal, oldpeak sesgada.

*En las variables categóricas, conté frecuencias y proporciones: 70% hombres, la mayoría con fbs normal, distribución equilibrada de exang, y variabilidad en cp, restecg, thal y slope.*

**3. Análisis bivariado**

Revisé correlaciones entre variables numéricas y con el target: thalach y oldpeak resultaron ser los más asociados al target, mientras que edad mostró relación negativa moderada.

*Usé heatmaps y pairplots para visualizar relaciones y outliers. Boxplots por target confirmaron que thalach y oldpeak son indicadores fuertes. Para las variables categóricas, hice tablas cruzadas y tests de chi-cuadrado, destacando cp, ca, thal y exang como las más predictivas del target, mientras que fbs no mostró relación significativa.*

**4. Análisis multivariado**

Para entender combinaciones de variables, revisé interacciones entre numéricas y categóricas con respecto al target usando pairplots, scatter plots y tablas cruzadas múltiples. Noté patrones interesantes:

- Pacientes más jóvenes con alta thalach y bajo oldpeak tenían más riesgo.

- Ciertas combinaciones de cp, thal y slope aumentaban claramente la probabilidad de enfermedad.

- Las relaciones entre variables numéricas y categóricas reforzaron la relevancia de las variables que ya habían destacado en el análisis bivariado.

*En general, este análisis me permitió entender bien la distribución de los datos, detectar relaciones importantes, confirmar qué variables podrían ser más relevantes para los modelos, y también identificar outliers y patrones que influirán en la ingeniería de características y el entrenamiento.*

## Ingeniería de Características (ft_engineering.py)

En este archivo me encargué de preparar los datos para el modelado, generando nuevas características, transformando variables y creando un pipeline completo que automatiza todo el proceso.

`Primero creé nuevas columnas derivadas para capturar relaciones entre variables, por ejemplo:`

- age_x_thalach = edad × frecuencia cardíaca máxima

- thalach_minus_age = frecuencia cardíaca máxima − edad

- oldpeak_ratio = depresión del ST / (thalach + 1e-6)

`Luego definí pipelines para procesar variables numéricas y categóricas:`

- Las numéricas se completan con la mediana y se escalan con StandardScaler.

- Las categóricas se completan con la moda y se codifican con one-hot encoding.

- Dividí los datos en conjuntos de entrenamiento y prueba (estratificando por el target) y apliqué un selector de características (SelectKBest) para mantener todas las variables, dejando la puerta abierta a futuras mejoras.

`Finalmente, guardé en mlops_pipeline/src/data:`

- La pipeline completa (pipeline_preprocessor.pkl)

- La lista de features (feature_list.txt)

- Los datasets de entrenamiento y prueba (train.csv y test.csv)

*Con esto, cualquier modelo que entrenemos podrá usar el mismo procesamiento de datos de forma consistente, asegurando que todas las transformaciones y nuevas variables se apliquen correctamente tanto en entrenamiento como en predicciones futuras.*

`Uso del script`
   ```
   python ft_engineering.py --input ../../Base_de_datos.csv --out_dir ./data --test_size 0.2 --random_state 42
   ```

## Entrenamiento y Evaluación de Modelos (model_training_evaluation.py)

En este archivo entrené y evalué distintos modelos supervisados usando los datos procesados en la etapa de ingeniería de características.

Primero cargué los datasets de entrenamiento y prueba (train.csv y test.csv) y apliqué el preprocesamiento guardado en ft_engineering.py para asegurar que todas las transformaciones y nuevas variables se aplicaran de manera consistente.

`Entrené dos modelos:` (REVISARRRRRRR)

- Logistic Regression

- Random Forest

Para cada modelo, calculé métricas de evaluación como accuracy, F1-score y ROC-AUC usando la función summarize_classification. Luego comparé los resultados y seleccioné el modelo con mejor F1-score como el modelo final.

`Guardé como salida:`

- El modelo seleccionado (best_model.pkl)

- Las métricas de todos los modelos (model_metrics.csv)

- Un gráfico comparativo de rendimiento (model_comparison.png)

*Con esto, tengo un modelo entrenado listo para ser usado en predicciones y puedo justificar la elección del mejor modelo basándome en métricas objetivas.*

`Uso del script:`
   ```
   python model_training_evaluation.py --train ./data/train.csv --test ./data/test.csv --out_dir ./data
   ```

## Monitoreo de Datos y Data Drift (model_monitoring.py)

En este archivo me encargué de revisar si los datos nuevos se desviaban de los datos de entrenamiento, lo que podría afectar el desempeño del modelo. Para esto, cargué los datasets de entrenamiento y prueba, y apliqué el mismo preprocesamiento que usamos para entrenar el modelo, asegurando que las transformaciones fueran consistentes.

`Calculé varias métricas para detectar Data Drift:`

- Para variables numéricas: PSI, KS test, Jensen-Shannon Divergence.

- Para variables categóricas: Chi-cuadrado.

- Además, etiqueté automáticamente los resultados con alertas de riesgo (🟢 OK, 🟡 Moderado, ⚠️ Alto) y marqué cuándo sería recomendable un retraining del modelo.

`Guardé los resultados en:`

- results.csv → resumen actual del monitoreo

- results_history/monitoring(fecha).csv → histórico con cada ejecución

*Con esto, puedo monitorear cambios en la distribución de los datos y detectar desviaciones que puedan afectar la predicción del modelo. La salida también permite tomar decisiones sobre cuándo actualizar o volver a entrenar el modelo.*

`Uso del script:`
   ```
   python model_monitoring.py
   ```

## Despliegue del Modelo (model_deploy.py)

En este archivo desplegué el modelo final usando FastAPI para poder realizar predicciones a través de una API.

`El flujo que implementé es el siguiente:`

- Carga del modelo (best_model.pkl) y del preprocesador (pipeline_preprocessor.pkl).

- Creación de un endpoint /predict que recibe un archivo CSV con nuevos datos y devuelve las predicciones en lote.

- Las predicciones se generan aplicando primero las transformaciones del preprocesador y las nuevas columnas derivadas definidas en ft_engineering.py.

`También incluí un endpoint / para comprobar que la API funciona correctamente.`

*Con esto, cualquier usuario puede subir un archivo con datos de pacientes y recibir predicciones sobre la probabilidad de enfermedad cardíaca, lo que facilita la integración del modelo en otras aplicaciones o procesos.*

`Uso de la API:`

- Levantar la API
   ```
   ejecutar el script con python model_deploy.py.
   ```

- Enviar un CSV al endpoint /predict para recibir las predicciones.

## Integración con SonarCloud

1. Para asegurar la calidad del código y la cobertura de pruebas, integré el proyecto con SonarCloud, conecté mi repositorio de GitHub con SonarQube Cloud, permitiendo analizar automáticamente el código y los tests.

2. Creé pruebas unitarias para los módulos principales (ft_engineering, model_deploy, model_monitoring, model_training_evaluation) dentro de mlops_pipeline/src/tests.

3. `Ejecuté las pruebas con cobertura usando:`
   ```
   pytest mlops_pipeline/tests --cov=mlops_pipeline/src --cov-report=xml
   ```

*Esto generó archivos de cobertura (coverage.xml y .coverage) que SonarCloud usa para evaluar el porcentaje de código probado.*

4. Subí el análisis completo a SonarCloud con:
   ```
   pysonar --sonar-token=<tu-token> --sonar-project-key=AlejoPerez10_ML_Proyecto_Final --sonar-organization=alejoperez10
   ```

*Con esto, puedo revisar métricas de calidad de código, vulnerabilidades, duplicaciones y cobertura de pruebas directamente en SonarCloud. Gracias a esta integración, garantizo que el código cumple con buenas prácticas y que las funciones principales están correctamente testeadas.*

## Despliegue Final en la Web (Docker + Render)

Después de asegurar la calidad del código y la cobertura de pruebas con SonarCloud, procedí a desplegar el modelo y la aplicación de manera que pudieran ser usados desde cualquier computadora o ubicación remota.

- `Preparación de Docker`

Creé dos Dockerfiles:

- Dockerfile.api → para desplegar la API del modelo (model_deploy.py).

- Dockerfile.streamlit → para desplegar la app de Streamlit que permite interactuar con el modelo de forma gráfica.

*Cada Dockerfile contiene todas las dependencias necesarias, el código fuente y la configuración del servidor (Uvicorn para FastAPI y Streamlit para la app).*

- `Construcción y prueba local`

Construí las imágenes Docker con los comandos:
   ```
   docker build -t myapi -f Dockerfile.api .
   ```
   ```
   docker build -t mystreamlit -f Dockerfile.streamlit .
   ```


- Probé los contenedores localmente para asegurar que la API y la app funcionaran correctamente antes del despliegue.

- Despliegue en Render

- Subí las imágenes Docker a Render, creando servicios separados para la API y la app.

- Esto permitió que la API y la app estén disponibles en la web, accesibles desde cualquier dispositivo sin necesidad de ejecutarlas localmente.

`Resultados`

- La API responde a solicitudes de predicción vía /predict y puede procesar archivos CSV en lote.

- La app de Streamlit permite visualizar resultados y métricas del modelo de forma interactiva y gráfica.

- Gracias a Docker y Render, aseguro disponibilidad, escalabilidad y facilidad de acceso para usuarios remotos.

*Con este paso, el proyecto queda completamente funcional y desplegado en la web, cumpliendo con todos los requisitos de accesibilidad y uso práctico.*

## Levantar conexión entre Docker, FastAPI y Streamlit `¡SOLO PARA ENTORNO LOCAL!`
1- Construir las imágenes Docker

- FastAPI
   ```
   docker build -t myapi -f mlops_pipeline/src/API/Dockerfile.api .
   ```

- Streamlit
   ```
   docker build -t mystreamlit -f mlops_pipeline/src/API/Dockerfile.streamlit .
   ```

2- Correr los contenedores
   
- FastAPI
   ```
   docker run -p 8000:8000 myapi
   ```

- Streamlit
   ```
   docker run -p 8501:8501 mystreamlit
   ```

3- Probar la APP
   • Streamlit: ```http://localhost:8501```
   • API FastAPI: ```http://localhost:8000```

4- Para detener los contenedores
   ```
   CTRL + C
   ```

## Ejecutar pruebas unitarias `(en local)`
   ```
   pytest mlops_pipeline/tests --cov=mlops_pipeline/src --cov-report=xml
   ```

## Ejecutar pruebas de SonarQube Cloud `(en local)`
   ```
   pysonar `--sonar-token=671bd2e4a569eb087980ba45285b40cc32db24d9 `--sonar-project-key=AlejoPerez10_ML_Proyecto_Final `--sonar-organization=alejoperez10
   ```