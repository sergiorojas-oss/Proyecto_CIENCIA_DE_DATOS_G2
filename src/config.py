"""Rutas, semilla y constantes compartidas por el pipeline y el dashboard."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS_CRUDOS = RAIZ / "data" / "raw"
DIR_DATOS_PROCESADOS = RAIZ / "data" / "processed"
DIR_ARTEFACTOS = RAIZ / "artefactos"

SEMILLA = 42

# versión del dataset fijada al mismo commit usado en el cuaderno de las Fases 1 a 3
COMMIT_DATASET = "8de46d135e859d9c0d5bc70f1d46c1d145db4e6d"
URL_DATASET = (
    "https://github.com/sergiorojas-oss/Proyecto_CIENCIA_DE_DATOS_G2/raw/"
    f"{COMMIT_DATASET}/Solicitudes%20Soportes.zip"
)
NOMBRE_ZIP = "Solicitudes_Soportes.zip"
NOMBRE_CSV = "Solicitudes Soportes.csv"

COLUMNAS_MATERIAL = ["Material_Servi_1", "Material_Servi_2", "Material_Servi_3", "Material_Servi_4"]

COLUMNAS_EXCLUIDAS = [
    "i.groupCode", "CATEGORIA", "SECCIONAL", "i.name", "ESTADO_ACT", "Serial_Equipo",
    "ID_SERVICE", "ESTADO", "ID_Codigo_AN", "CC", "i.externalid", "DIRECCION",
]

GRUPOS_FALLA = {
    "Sin senal o nivel bajo": "Conectividad", "Reconfiguracion de IP": "Conectividad",
    "POS no carga": "Conectividad", "Cable de red": "Conectividad",
    "Dano o bloqueo": "Hardware", "Falla del pin": "Hardware",
    "Datafono no enciende": "Hardware", "Tamper bloqueo": "Hardware",
    "Actualizacion": "Software y transacciones", "tarjeta rechazada": "Software y transacciones",
    "Mantenimiento preventivo": "Preventivo", "SOPORTE PROACTIVO": "Preventivo",
}

VARIABLES_CATEGORICAS = ["NOMBRE_EQUIPO", "TECNOLOGIA", "CIUDAD", "CENTRO_Y_AL"]
VARIABLES_NUMERICAS = [
    "es_android", "es_inalambrico", "mes_sin", "mes_cos", "dia_semana",
    "es_fin_de_semana", "temporada_alta", "tickets_previos_comercio", "dias_desde_ultimo_ticket",
]
ENTRADAS_TIPO = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS
ENTRADAS_REPUESTO = VARIABLES_CATEGORICAS + ["TIPO_PROBLEMA"] + VARIABLES_NUMERICAS
ENTRADAS_REGRESION = VARIABLES_CATEGORICAS + ["TIPO_PROBLEMA", "Tecnico"] + VARIABLES_NUMERICAS

# hiperparámetros seleccionados por validación cruzada en el cuaderno (secciones 8.1, 9.1 y 9.3)
PARAMS_REPUESTO = {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 25, "max_features": 0.3}
PARAMS_TIPO = {"n_estimators": 150, "max_depth": 14, "min_samples_leaf": 50, "class_weight": None}
PARAMS_REGRESION = {"max_depth": 8, "learning_rate": 0.1, "min_samples_leaf": 20, "l2_regularization": 1.0}

INDICADORES_COMERCIO = [
    "tickets_mes", "equipos", "pct_conectividad", "pct_hardware", "pct_preventivo",
    "pct_requiere_repuesto", "dias_resolucion_prom", "pct_inalambrico",
]
K_SEGMENTOS = 3
ORDEN_SEGMENTOS = ["Grandes cuentas", "Hardware intensivo", "Conectividad inalámbrica"]

ESTRATEGIAS_SEGMENTO = {
    "Grandes cuentas": "Visitas programadas y acuerdo de nivel de servicio dedicado; stock de repuestos en sitio.",
    "Hardware intensivo": "Revisión física periódica del terminal (teclado, pin pad, batería) y stock local de repuestos.",
    "Conectividad inalámbrica": "Soporte remoto de primer nivel (señal, SIM, IP) antes de despachar un técnico.",
}
