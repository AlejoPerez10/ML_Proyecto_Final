# 🧠 Proyecto de Modelo Supervisado Predictivo

## 📁 Estructura del Proyecto

```
MACHINE-LEARNING/
│
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb         # Carga y preprocesamiento inicial de los datos
│       ├── comprension_eda.ipynb      # Análisis exploratorio de datos (EDA)
│       ├── ft_engineering.py          # Ingeniería de características
│       ├── heuristic_model.py         # Modelo base o heurístico para comparación
│       ├── model_training.ipynb       # Entrenamiento del modelo
│       ├── model_evaluation.ipynb     # Evaluación y métricas del modelo
│       ├── model_deploy.ipynb         # Despliegue del modelo
│       ├── model_monitoring.ipynb     # Monitoreo del modelo en producción
│
├── Base_de_datos.csv                  # Fuente principal de datos
├── config.json                        # Configuraciones globales del proyecto
├── requirements.txt                   # Librerías necesarias
├── set_up.bat                         # Script para entorno de ejecución en Windows
├── readme.md                          # Este archivo :)
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

## 📊 Métricas de Evaluación
- **MAE (Error Absoluto Medio)**  
- **MSE (Error Cuadrático Medio)**  
- **RMSE (Raíz del Error Cuadrático Medio)**  
- **R² (Coeficiente de Determinación)**  

## 🚀 Cómo Ejecutarlo
1. Clonar este repositorio  
   ```bash
   git clone https://github.com/tu-usuario/proyecto-ml-regresion.git

