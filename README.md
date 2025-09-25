# PREDICTOR-DE-PRECIO-DE-SEGURO-M-DICO
Sistema de machine learning para predecir costos de seguros médicos. Desarrollado con Python, incluye análisis exploratorio, ingeniería de features y comparación de modelos. Alcanza 87.8% de precisión (R²)
# 🏥 Insurance Cost Predictor - Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Descripción del Proyecto

Sistema completo de **Machine Learning** para predecir costos de seguros médicos basado en características demográficas y de salud. Este proyecto demuestra un flujo completo de **Data Science** desde el análisis exploratorio hasta el deployment del modelo.

## 🎯 Resultados Destacados

| Métrica | Valor | Explicación |
|---------|-------|-------------|
| **R² Score** | 0.8781 | El modelo explica el 87.81% de la varianza |
| **RMSE** | $4,349 | Error promedio de predicción |
| **MAE** | $1,935 | Error absoluto promedio |

## 🚀 Características Principales

### 🔍 Análisis Exploratorio Avanzado
- Distribución de costos por edad, género y región
- Impacto del tabaquismo en los costos médicos
- Análisis de correlaciones y valores atípicos

### ⚙️ Ingeniería de Features
- Codificación de variables categóricas
- Creación de features de interacción (edad × fumador)
- Transformación logarítmica del target
- Categorización de BMI y grupos de edad

### 🤖 Modelos de Machine Learning Comparados
| Modelo | R² Score | RMSE |
|--------|----------|------|
| Gradient Boosting | 0.8781 | 4,349 |
| Random Forest | 0.8715 | 4,466 |
| Ridge Regression | 0.8523 | 4,788 |
| Linear Regression | 0.8490 | 4,841 |
| XGBoost | 0.8101 | 5,429 |

### 📈 Insights Clave
- **Fumar es el factor más crítico** (33% de importancia)
- La **edad multiplica el efecto** de ser fumador
- Combinación **BMI alto + fumador** genera costos elevados
- **Región sureste** muestra costos promedio más altos

## 🛠️ Instalación y Uso

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/insurance-cost-predictor.git
cd insurance-cost-predictor

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el notebook principal
jupyter notebook insurance_cost_analysis.ipynb
```

## 📁 Estructura del Proyecto

```
insurance-cost-predictor/
│
├── data/
│   └── insurance.csv              
├── models/
│   ├── insurance_cost_predictor.pkl  
│   ├── scaler.pkl                
│   └── model_metadata.json        
├── insurance_cost_analysis.ipynb  
├── requirements.txt               
└── README.md                      
```

## 💡 Uso del Modelo

```python
import joblib

# Cargar modelo y scaler
model = joblib.load('models/insurance_cost_predictor.pkl')
scaler = joblib.load('models/scaler.pkl')

# Realizar predicción
prediccion = model.predict(datos_nuevos)
costo_estimado = np.expm1(prediccion)
```

## 🎨 Visualizaciones Incluidas

- 📊 Dashboard interactivo con Plotly
- 🔥 Matriz de correlaciones
- 📈 Análisis de residuales
- 🎯 Importancia de variables
- 📉 Distribuciones y outliers

## 🏆 Habilidades Demostradas

- **Preprocesamiento de datos** con Pandas y NumPy
- **Visualización** con Matplotlib, Seaborn y Plotly
- **Machine Learning** con Scikit-learn y XGBoost
- **Optimización** con GridSearchCV
- **Feature Engineering** avanzado
- **Análisis de modelos** y métricas

## 📊 Dataset

El dataset contiene **1,338 registros** con las siguientes variables:
- `age`: Edad del beneficiario
- `sex`: Género
- `bmi`: Índice de masa corporal
- `children`: Número de hijos/dependientes
- `smoker`: ¿Es fumador?
- `region`: Región de residencia
- `charges`: Costos médicos individuales

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

⭐ **Si este proyecto te resultó útil, por favor dale una estrella en GitHub!**
