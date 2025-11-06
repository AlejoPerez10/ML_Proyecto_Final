# 🧠 Proyecto de Modelo Supervisado Predictivo

## 📁 Estructura del Proyecto

```
MACHINE-LEARNING/
├── entorno_ml-venv/                         # Entorno virtual con todas las librerías necesarias.
├── mlops_pipeline/
│   └── src/
│       ├── __pycache__/                     # Archivos compilados de Python para rápida ejecución.
│       ├── data/                            # almacena los datasets y pruebas del modelo.
│       ├── Cargar_datos.ipynb               # Carga y preprocesamiento inicial de los datos
│       ├── comprension_eda.ipynb            # Análisis exploratorio de datos (EDA)
│       ├── ft_engineering.py                # Ingeniería de características
│       ├── heuristic_model.py               # Modelo base o heurístico para comparación
│       ├── model_deploy.py                  # Despliegue del modelo
│       ├── model_evaluation.ipynb           # Evaluación del modelo
│       ├── model_monitoring.py              # Entrenamiento del modelo
│       ├── model_training_evaluation.py     # Monitoreo del modelo en producción
│
├── Base_de_datos.csv                        # Fuente principal de datos
├── config.json                              # Configuraciones globales del proyecto
├── readme.md                                # Este archivo
├── requirements.txt                         # Librerías necesarias
├── set_up.bat                               # Script para entorno de ejecución en Windows
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

## Columnas de mi base de datos

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

## Categorías de mis variables

• **Numéricas →** Son cantidades medibles. Se pueden sumar o promediar.
Ej: edad, presión, colesterol, frecuencia cardíaca, oldpeak → como medir peso o temperatura.

• **Categóricas →** Representan grupos o etiquetas, no cantidades.
Ej: sex (h/m), fbs (sí/no), exang (sí/no), target (enfermo/sano), ca (0–3 vasos).

• **Categóricas nominales →** Son categorías sin orden natural.
Ej: cp, restecg, thal → tipos distintos (no mejores ni peores entre sí).

• **Categórica ordinal →** Son categorías con orden lógico.
Ej: slope (0 ascendente, 1 plana, 2 descendente).


