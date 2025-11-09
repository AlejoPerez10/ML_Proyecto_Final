# 🧠 Proyecto de Modelo Supervisado Predictivo

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

## 🐍 Activación del Entorno Virtual
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


## 📋 Descripción
Este proyecto aplica técnicas de **regresión en Machine Learning** para predecir una variable continua a partir de un conjunto de datos.  
Incluye todo el proceso de limpieza, entrenamiento, evaluación y visualización de resultados.

## ⚙️ Características
- Limpieza y normalización de datos  
- Selección de variables y análisis de correlaciones  
- Entrenamiento de diferentes modelos de regresión  
- Evaluación del rendimiento con métricas estadísticas  
- Visualización de resultados y análisis de errores  

## 🧩 Modelos Utilizados
- Regresión Lineal  
- Árbol de Decisión Regressor  
- Random Forest Regressor  
- Support Vector Regressor (SVR)  

## Columnas de la base de datos

**1-** age

**2-** sex

**3-** chest pain type (4 values)
   0 - Angina Típica
   1 - Angina Atípica
   2 - Dolor no angionoso
   3 - Asintómatico

**4-** resting blood pressure
   Presión arterial en reposo

**5-** serum cholestoral in mg/dl
   Colesteron sérico

**6-** fasting blood sugar > 120 mg/dl
   Azúcar en ayunas

**7-** resting electrocardiographic results (values 0,1,2)
   0 - Normal
   1 - Anomalía de la onda ST-T
   2 - Hipertrofia ventricular izquierda

**8-** maximum heart rate achieved
   Frecuencia cardiaca máxima aclanzada

**9-** exercise induced angina
   Angina inducida por ejercicio

**10-** oldpeak = ST depression induced by exercise relative to rest
   Depresión del segmenteo ST provocado por ejercicio

**11-** the slope of the peak exercise ST segment
   Pendiente del segmento ST
   0 - ascendente
   1 - plano
   2 - descendente

**12-** number of major vessels (0-3) colored by flourosopy
   Número de vasos principales (observados por fluoroscopia)

**13-** thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
   Tipo de talsemio o defecto sanguíneo
   0 - normal
   1 - defecto fijo (permanente)
   2 - defecto reversible (mejora con esfuerzo o tratamiento)

**14-** Target 
   0 - (NO) Enfermedad cardiaca ausente
   1 - (SÍ) Enfermedad cardiaca presente

## Categorías de las variables

• **Numéricas →** Son cantidades medibles. Se pueden sumar o promediar.
Ej: edad, presión, colesterol, frecuencia cardíaca, oldpeak → como medir peso o temperatura.

• **Categóricas →** Representan grupos o etiquetas, no cantidades.
Ej: sex (h/m), fbs (sí/no), exang (sí/no), target (enfermo/sano), ca (0–3 vasos).

• **Categóricas nominales →** Son categorías sin orden natural.
Ej: cp, restecg, thal → tipos distintos (no mejores ni peores entre sí).

• **Categórica ordinal →** Son categorías con orden lógico.
Ej: slope (0 ascendente, 1 plana, 2 descendente).

## Model Monitoring
• Al analizar los resultados del monitoreo del modelo, se observa que la mayoría de las variables se mantienen estables entre los datos de entrenamiento y los de prueba, lo que indica que el modelo sigue recibiendo información similar a la que fue entrenado.

• Sin embargo, la variable “chol” (colesterol) muestra un ligero cambio (PSI 0.11 y KS p 0.013), lo que sugiere una pequeña diferencia en la distribución de los datos nuevos. Esto no afecta gravemente el desempeño, pero sí vale la pena seguir revisándola en futuras ejecuciones para asegurarse de que el modelo no empiece a degradarse.

• En general, el modelo está estable y sin señales de drift importantes, lo que significa que por ahora se puede seguir usando sin necesidad de reentrenarlo.

## Levantar conexión entre Docker, FastAPI y Streamlit
1- Construir las imágenes Docker

   • FastAPI
   ```
   docker build -t myapi -f mlops_pipeline/src/API/Dockerfile.api .
   ```

   • Streamlit
   ```
   docker build -t mystreamlit -f mlops_pipeline/src/API/Dockerfile.streamlit .
   ```

2- Correr los contenedores
   
   • FastAPI
   ```
   docker run -p 8000:8000 myapi
   ```

   • Streamlit
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

## Ejecutar pruebas de SonarQube Cloud
   ```
   pysonar `--sonar-token=671bd2e4a569eb087980ba45285b40cc32db24d9 `--sonar-project-key=AlejoPerez10_ML_Proyecto_Final `--sonar-organization=alejoperez10
   ```

## Pruebas en Sonar
![Primera prueba en SonarQube Cloud](/images/image.png)

## Ejecutar pruebas unitarias
   ```
   pytest mlops_pipeline/tests --cov=mlops_pipeline/src --cov-report=xml
   ```