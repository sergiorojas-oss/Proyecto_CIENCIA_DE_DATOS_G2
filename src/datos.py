"""Descarga, limpieza e ingeniería de variables.

Reproduce las secciones 4 y 5 del cuaderno de las Fases 1 a 3: eliminación de duplicados,
columnas sin valor analítico, fechas con formato explícito, imputación determinística del
estado y variables derivadas del dominio.
"""
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src import config


def descargar_dataset(forzar=False):
    """Descarga el ZIP del repositorio (commit fijado) y devuelve la ruta del CSV."""
    config.DIR_DATOS_CRUDOS.mkdir(parents=True, exist_ok=True)
    ruta_csv = config.DIR_DATOS_CRUDOS / config.NOMBRE_CSV
    if ruta_csv.exists() and not forzar:
        return ruta_csv
    ruta_zip = config.DIR_DATOS_CRUDOS / config.NOMBRE_ZIP
    if not ruta_zip.exists() or forzar:
        print("Descargando el dataset desde GitHub (commit fijado)...")
        urllib.request.urlretrieve(config.URL_DATASET, ruta_zip)
    with zipfile.ZipFile(ruta_zip) as archivo:
        archivo.extractall(config.DIR_DATOS_CRUDOS)
    return ruta_csv


def cargar_crudos(ruta_csv=None):
    ruta_csv = ruta_csv or descargar_dataset()
    return pd.read_csv(ruta_csv, sep=";", low_memory=False)


def preparar(df):
    """Limpieza y variables derivadas. Devuelve un DataFrame ordenado por fecha de apertura."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=[c for c in config.COLUMNAS_EXCLUIDAS if c in df.columns])

    df["CREADO_EN_dt"] = pd.to_datetime(df["CREADO_EN"].astype(str).str.rstrip("T"),
                                        format="%Y-%m-%d", errors="coerce")
    df["FINALIZADO_dt"] = pd.to_datetime(df["FINALIZADO"], format="%d/%m/%Y", errors="coerce")
    df["cierre_incoherente"] = (df["FINALIZADO_dt"] < df["CREADO_EN_dt"]).astype(int)

    # el estado faltante se reconstruye con la fecha de cierre (imputación determinística)
    sin_estado = df["Estado"].isna()
    df.loc[sin_estado & df["FINALIZADO_dt"].notna(), "Estado"] = "Cerrado"
    df["Estado"] = df["Estado"].fillna("Sin registro")
    df["CENTRO_Y_AL"] = df["CENTRO_Y_AL"].fillna("Sin registro")

    # códigos de repuesto dañados por la notación científica de Excel
    for columna in config.COLUMNAS_MATERIAL:
        df[columna] = df[columna].replace(r"^\d+,\d+E\+\d+$", "Ilegible", regex=True)

    df = df.sort_values(["CREADO_EN_dt", "ID_Llamada"]).reset_index(drop=True)
    df["grupo_falla"] = df["TIPO_PROBLEMA"].map(config.GRUPOS_FALLA)

    materiales = df[config.COLUMNAS_MATERIAL]
    df["num_repuestos"] = (materiales.notna() & (materiales != "Ilegible")).sum(axis=1)
    df["requiere_repuesto"] = materiales.notna().any(axis=1).astype(int)
    df["repuesto_principal"] = df["Material_Servi_1"].fillna("Ninguno")

    df = agregar_variables_calendario_y_equipo(df)

    # historial del comercio calculado solo con tickets anteriores (sin fuga de información)
    df["tickets_previos_comercio"] = df.groupby("USUARIO").cumcount()
    df["dias_desde_ultimo_ticket"] = df.groupby("USUARIO")["CREADO_EN_dt"].diff().dt.days.fillna(-1)

    cierre_valido = df["FINALIZADO_dt"].notna() & (df["cierre_incoherente"] == 0)
    df["dias_resolucion"] = np.where(cierre_valido,
                                     (df["FINALIZADO_dt"] - df["CREADO_EN_dt"]).dt.days, np.nan)
    return df


def agregar_variables_calendario_y_equipo(df):
    """Variables que dependen solo del terminal y de la fecha; se usan también en la predicción en vivo."""
    df = df.copy()
    mes = df["CREADO_EN_dt"].dt.month
    df["es_android"] = df["NOMBRE_EQUIPO"].astype(str).str.contains("ANDROID").astype(int)
    df["es_inalambrico"] = df["TECNOLOGIA"].isin(["GPRS", "Wi-Fi"]).astype(int)
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    df["dia_semana"] = df["CREADO_EN_dt"].dt.dayofweek
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["temporada_alta"] = mes.isin([12, 1]).astype(int)
    return df


def construir_ticket(equipo, tecnologia, ciudad, centro, tipo_problema, tecnico, fecha,
                     tickets_previos=0, dias_desde_ultimo=-1):
    """Arma un ticket nuevo con las mismas columnas que reciben los modelos."""
    ticket = pd.DataFrame([{
        "NOMBRE_EQUIPO": equipo, "TECNOLOGIA": tecnologia, "CIUDAD": ciudad, "CENTRO_Y_AL": centro,
        "TIPO_PROBLEMA": tipo_problema, "Tecnico": tecnico, "CREADO_EN_dt": pd.Timestamp(fecha),
        "tickets_previos_comercio": int(tickets_previos), "dias_desde_ultimo_ticket": float(dias_desde_ultimo),
    }])
    return agregar_variables_calendario_y_equipo(ticket)


def historial_comercio(tickets, usuario, fecha):
    """Tickets previos del comercio y días desde su último ticket antes de `fecha`."""
    previos = tickets.loc[(tickets["USUARIO"] == usuario) & (tickets["CREADO_EN_dt"] < pd.Timestamp(fecha)),
                          "CREADO_EN_dt"]
    if previos.empty:
        return 0, -1
    return len(previos), (pd.Timestamp(fecha) - previos.max()).days
