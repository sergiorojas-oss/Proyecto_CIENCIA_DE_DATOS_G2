# Datos

- `raw/`: aquí se descarga y descomprime `Solicitudes Soportes.zip` (commit `8de46d1` del repositorio). No se versiona; `python -m src.entrenar` lo descarga automáticamente.
- `processed/`: archivos generados por `python -m src.entrenar` y consumidos por el dashboard:
  - `tickets_dashboard.csv.gz`: 209,160 tickets depurados con las variables usadas en filtros y KPIs.
  - `segmentacion_comercios.csv`: indicadores de los 900 comercios, segmento K-Means y Ward, y coordenadas PCA.
  - `demanda_semanal.csv`: serie semanal por ciudad, pronóstico del periodo de prueba y de la semana siguiente.
