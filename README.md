# Analítica predictiva para la anticipación de fallas y la planeación logística del soporte de datáfonos en campo

Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial · Maestría en Inteligencia Artificial, Universidad de La Salle
Curso CIENCIA DE DATOS_G2 · Docente: Fabián Camilo Castro Riveros · Autor: Sergio Andrés Rojas Ordóñez

El proyecto traslada cuatro decisiones de la operación de soporte de datáfonos al momento en que se abre el ticket:

| Decisión operativa | Modelo | Resultado en prueba |
| :--- | :--- | :--- |
| ¿La visita requiere repuesto? | Random Forest | AUC-ROC 0.842, F1 0.797 |
| ¿Qué fallas cargar en el kit del técnico? | Random Forest (top-3) | 62.4 % frente a 39.5 % de la referencia |
| ¿Cuántos días comprometer con el comercio? | Gradient Boosting | MAE 2.98 días frente a 4.53 del Dummy |
| ¿Cuántos tickets llegarán a cada ciudad? | Ridge con rezagos | MAE 18.8 tickets frente a 21.4 del ingenuo |
| ¿Qué comercios atender de forma preventiva? | K-Means (k = 3) | Silueta 0.314 |

## Estructura del repositorio

```
├── data/
│   ├── raw/                 dataset original (se descarga desde el commit 8de46d1)
│   └── processed/           tickets depurados, segmentos y serie semanal para el dashboard
├── notebooks/
│   └── Proyecto_Integrador_Fases_1_2_3.ipynb   EDA, preprocesamiento, modelado, MLP, SHAP
├── src/
│   ├── config.py            rutas, semilla, variables e hiperparámetros seleccionados
│   ├── datos.py             descarga, limpieza e ingeniería de variables
│   └── entrenar.py          entrenamiento de los modelos finales y exportación de artefactos
├── dashboard/
│   ├── app.py               aplicación Streamlit
│   └── requirements.txt     dependencias del despliegue
├── artefactos/              modelos entrenados (.joblib) y metricas.json
├── .streamlit/config.toml   tema del dashboard
└── requirements.txt         entorno completo (pipeline, dashboard y cuaderno)
```

## Ejecución local

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/sergiorojas-oss/Proyecto_CIENCIA_DE_DATOS_G2.git
cd Proyecto_CIENCIA_DE_DATOS_G2
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m src.entrenar           # descarga los datos, entrena y guarda artefactos (2 a 4 minutos)
streamlit run dashboard/app.py   # abre el dashboard en http://localhost:8501
```

Si `artefactos/` está vacío, el dashboard ejecuta el entrenamiento automáticamente la primera vez.

## Dashboard

| Módulo | Contenido |
| :--- | :--- |
| Filtros dinámicos (barra lateral) | Rango de fechas, ciudad, tecnología, modelo de equipo, grupo de falla y rango de días de resolución |
| Resumen operativo | Tarjetas KPI de la operación y de los modelos; tickets por semana, ciudad y tipo de falla; días de resolución; repuestos más usados |
| Predicción en vivo | Simulador de un ticket nuevo: probabilidad de repuesto con umbral de despacho ajustable, repuestos sugeridos, días estimados y kit de las tres fallas más probables |
| Segmentación de comercios | Mapa de clústeres en el espacio PCA (2D y 3D), K-Means o Ward, perfiles promedio, estrategia por segmento, codo y silueta |
| Demanda semanal | Serie real y pronosticada por ciudad y pronóstico de la semana siguiente |
| Desempeño de los modelos | Métricas de validación y prueba frente a las metas de los objetivos específicos |

## Despliegue en Streamlit Community Cloud

1. Ejecute `python -m src.entrenar` y suba al repositorio las carpetas `artefactos/` y `data/processed/`.
2. En https://share.streamlit.io elija **Create app**, seleccione este repositorio, la rama `main` y el archivo principal `dashboard/app.py`.
3. En **Advanced settings** elija Python 3.12 y despliegue. Streamlit Cloud instala `dashboard/requirements.txt`.

## Reproducibilidad

- Semilla única: 42. Partición estratificada 70/15/15 por tipo de falla.
- El dataset se descarga siempre del commit `8de46d135e859d9c0d5bc70f1d46c1d145db4e6d`.
- Los hiperparámetros de `src/config.py` son los seleccionados por validación cruzada en el cuaderno (secciones 8.1, 9.1 y 9.3).
- Los modelos `.joblib` deben cargarse con la misma versión de scikit-learn con la que se entrenaron (1.9.1).

Cuaderno en Google Colab: https://colab.research.google.com/drive/18XEVdF7Eli72OyFBsnA5iUvLKbqz_RTj

## Nota sobre los datos

El conjunto de datos es sintético: recrea la estructura de una operación real de soporte de datáfonos (modelos de terminal, tecnologías, tipos de falla, repuestos, ciudades y técnicos) con comercios, técnicos y seriales anonimizados. Los resultados deben validarse con datos de la operación antes de implementarse.
