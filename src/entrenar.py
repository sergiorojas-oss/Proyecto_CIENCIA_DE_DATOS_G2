"""Entrena los modelos finales y guarda los artefactos que consume el dashboard.

Uso (desde la raíz del repositorio):
    python -m src.entrenar

Reproduce la partición estratificada 70/15/15 y los modelos seleccionados en el cuaderno
(la búsqueda de hiperparámetros completa está en notebooks/). Genera:
    artefactos/modelo_requiere_repuesto.joblib   Random Forest (¿la visita requiere repuesto?)
    artefactos/modelo_tipo_falla.joblib          Random Forest (kit de las 3 fallas más probables)
    artefactos/modelo_dias_resolucion.joblib     Gradient Boosting (días de resolución)
    artefactos/modelo_demanda.joblib             Ridge con rezagos (tickets por semana y ciudad)
    artefactos/metricas.json                     métricas de validación y prueba
    data/processed/tickets_dashboard.csv.gz      tickets depurados para filtros y KPIs
    data/processed/segmentacion_comercios.csv    indicadores, segmentos y coordenadas PCA
    data/processed/demanda_semanal.csv           serie semanal, pronóstico de prueba y próxima semana
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, silhouette_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.datos import cargar_crudos, preparar

S = config.SEMILLA


def preprocesador(categoricas):
    return ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ("numericas", StandardScaler(), config.VARIABLES_NUMERICAS),
    ])


def metricas_binarias(y, p, umbral=0.5):
    pred = (p >= umbral).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "AUC-ROC": roc_auc_score(y, p)}


def metricas_regresion(y, pred):
    return {"R2": r2_score(y, pred), "MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred)))}


def particionar(df):
    idx = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, stratify=df["TIPO_PROBLEMA"], random_state=S)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=df["TIPO_PROBLEMA"].iloc[idx_tmp],
                                      random_state=S)
    return df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]


def entrenar_repuesto(train, val, test):
    X_tr, y_tr = train[config.ENTRADAS_REPUESTO], train["requiere_repuesto"]
    pre = ["NOMBRE_EQUIPO", "TECNOLOGIA", "CIUDAD", "CENTRO_Y_AL", "TIPO_PROBLEMA"]
    candidatos = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=S, n_jobs=-1, **config.PARAMS_REPUESTO),
    }
    tabla, modelos = {}, {}
    for nombre, estimador in candidatos.items():
        modelo = Pipeline([("pre", preprocesador(pre)), ("modelo", estimador)]).fit(X_tr, y_tr)
        modelos[nombre] = modelo
        tabla[nombre] = metricas_binarias(val["requiere_repuesto"],
                                          modelo.predict_proba(val[config.ENTRADAS_REPUESTO])[:, 1])
    rf = modelos["Random Forest"]
    p_test = rf.predict_proba(test[config.ENTRADAS_REPUESTO])[:, 1]
    y_test = test["requiere_repuesto"].to_numpy()
    politica = []
    for umbral in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        lleva = p_test >= umbral
        politica.append({"umbral": umbral,
                         "pct_despachadas_con_repuesto": float(lleva.mean() * 100),
                         "cobertura_recall": float(recall_score(y_test, lleva) * 100),
                         "uso_precision": float(precision_score(y_test, lleva, zero_division=0) * 100),
                         "visitas_sin_cubrir": int(((~lleva) & (y_test == 1)).sum())})
    return rf, {"validacion": tabla, "prueba": metricas_binarias(y_test, p_test), "politica": politica,
                "tasa_base": float(y_test.mean())}


def entrenar_tipo(train, val, test):
    pre = config.VARIABLES_CATEGORICAS
    modelo = Pipeline([("pre", preprocesador(pre)),
                       ("modelo", RandomForestClassifier(random_state=S, n_jobs=-1, **config.PARAMS_TIPO))])
    modelo.fit(train[config.ENTRADAS_TIPO], train["TIPO_PROBLEMA"])
    clases = modelo.classes_
    prior = train["TIPO_PROBLEMA"].value_counts(normalize=True).reindex(clases).to_numpy()
    resultado = {}
    for nombre, conjunto in [("validacion", val), ("prueba", test)]:
        prob = modelo.predict_proba(conjunto[config.ENTRADAS_TIPO])
        y = conjunto["TIPO_PROBLEMA"]
        resultado[nombre] = {
            "Accuracy": accuracy_score(y, clases[prob.argmax(axis=1)]),
            "F1 macro": f1_score(y, clases[prob.argmax(axis=1)], average="macro"),
            "Top-3 accuracy": top_k_accuracy_score(y, prob, k=3, labels=clases),
            "Top-3 referencia": top_k_accuracy_score(y, np.tile(prior, (len(y), 1)), k=3, labels=clases),
        }
    return modelo, resultado


def entrenar_regresion(train, val, test):
    conjuntos = [c.dropna(subset=["dias_resolucion"]).copy() for c in (train, val, test)]
    limite = float(conjuntos[0]["dias_resolucion"].quantile(0.99))   # percentil 99 solo de entrenamiento
    for c in conjuntos:
        c["y"] = c["dias_resolucion"].clip(upper=limite)
    tr, va, te = conjuntos
    pre = ["NOMBRE_EQUIPO", "TECNOLOGIA", "CIUDAD", "CENTRO_Y_AL", "TIPO_PROBLEMA", "Tecnico"]
    modelo = Pipeline([("pre", preprocesador(pre)),
                       ("modelo", HistGradientBoostingRegressor(random_state=S, **config.PARAMS_REGRESION))])
    modelo.fit(tr[config.ENTRADAS_REGRESION], tr["y"])
    media = tr["y"].mean()
    resultado = {"limite_winsorizacion": limite}
    for nombre, c in [("validacion", va), ("prueba", te)]:
        resultado[nombre] = metricas_regresion(c["y"], modelo.predict(c[config.ENTRADAS_REGRESION]))
        resultado[nombre + "_dummy"] = metricas_regresion(c["y"], np.full(len(c), media))
    residuo_abs = np.abs(te["y"] - modelo.predict(te[config.ENTRADAS_REGRESION]))
    resultado["pct_error_menor_3_dias"] = float((residuo_abs < 3).mean() * 100)
    return modelo, resultado


def serie_semanal(df):
    semanal = (df.groupby([pd.Grouper(key="CREADO_EN_dt", freq="W-SUN"), "CIUDAD"]).size()
                 .rename("tickets").reset_index().rename(columns={"CREADO_EN_dt": "semana"}))
    semanal = semanal[semanal["semana"] <= df["CREADO_EN_dt"].max()]   # se descarta la semana incompleta
    return semanal.sort_values(["CIUDAD", "semana"]).reset_index(drop=True)


def variables_demanda(semanal):
    s = semanal.copy()
    grupo = s.groupby("CIUDAD")["tickets"]
    for rezago in (1, 2, 3, 4):
        s[f"rezago_{rezago}"] = grupo.shift(rezago)
    s["media_movil_4"] = grupo.transform(lambda x: x.shift(1).rolling(4).mean())
    s["temporada_alta"] = s["semana"].dt.month.isin([12, 1]).astype(int)
    return s


ENTRADAS_DEMANDA = ["rezago_1", "rezago_2", "rezago_3", "rezago_4", "media_movil_4", "temporada_alta"]


def matriz_demanda(s, ciudades):
    X = s[ENTRADAS_DEMANDA].astype(float).copy()
    for ciudad in ciudades:
        X[f"CIUDAD_{ciudad}"] = (s["CIUDAD"] == ciudad).astype(float)
    return X


def entrenar_demanda(df):
    semanal = variables_demanda(serie_semanal(df))
    ciudades = sorted(semanal["CIUDAD"].unique())
    datos = semanal.dropna().reset_index(drop=True)
    X = matriz_demanda(datos, ciudades)
    corte_val, corte_test = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-07-01")
    es_trva = datos["semana"] < corte_test
    es_te = ~es_trva
    nuevo_ridge = lambda: Pipeline([("escala", StandardScaler()), ("modelo", Ridge(alpha=1.0))])

    modelo_eval = nuevo_ridge().fit(X[es_trva], datos.loc[es_trva, "tickets"])
    datos["pronostico"] = np.nan
    datos.loc[es_te, "pronostico"] = modelo_eval.predict(X[es_te])
    resultado = {"prueba": metricas_regresion(datos.loc[es_te, "tickets"], datos.loc[es_te, "pronostico"]),
                 "prueba_ingenuo": metricas_regresion(datos.loc[es_te, "tickets"], datos.loc[es_te, "rezago_1"]),
                 "corte_validacion": str(corte_val.date()), "corte_prueba": str(corte_test.date())}

    # modelo final con toda la historia para pronosticar la semana siguiente a la última observada
    modelo_final = nuevo_ridge().fit(X, datos["tickets"])
    siguiente = []
    for ciudad in ciudades:
        hist = semanal[semanal["CIUDAD"] == ciudad].sort_values("semana")
        ultimos = hist["tickets"].to_numpy()[-4:][::-1]
        semana = hist["semana"].max() + pd.Timedelta(weeks=1)
        fila = {"semana": semana, "CIUDAD": ciudad, **{f"rezago_{i + 1}": ultimos[i] for i in range(4)},
                "media_movil_4": ultimos.mean(), "temporada_alta": int(semana.month in (12, 1))}
        siguiente.append(fila)
    siguiente = pd.DataFrame(siguiente)
    siguiente["pronostico"] = modelo_final.predict(matriz_demanda(siguiente, ciudades))
    siguiente["tickets"] = np.nan
    siguiente["tipo"] = "Próxima semana"
    datos["tipo"] = np.where(es_te, "Prueba", "Entrenamiento")
    serie = pd.concat([datos[["semana", "CIUDAD", "tickets", "pronostico", "tipo"]],
                       siguiente[["semana", "CIUDAD", "tickets", "pronostico", "tipo"]]], ignore_index=True)
    return {"modelo": modelo_final, "ciudades": ciudades}, resultado, serie


def segmentar_comercios(df):
    meses = (df["CREADO_EN_dt"].max() - df["CREADO_EN_dt"].min()).days / 30.44
    tabla = df.groupby("USUARIO").agg(
        tickets_mes=("ID_Llamada", lambda s: len(s) / meses),
        equipos=("Serial_Actuac", "nunique"),
        pct_conectividad=("grupo_falla", lambda s: (s == "Conectividad").mean() * 100),
        pct_hardware=("grupo_falla", lambda s: (s == "Hardware").mean() * 100),
        pct_preventivo=("grupo_falla", lambda s: (s == "Preventivo").mean() * 100),
        pct_requiere_repuesto=("requiere_repuesto", lambda s: s.mean() * 100),
        dias_resolucion_prom=("dias_resolucion", "mean"),
        pct_inalambrico=("es_inalambrico", lambda s: s.mean() * 100),
        ciudad_principal=("CIUDAD", lambda s: s.mode().iloc[0]),
        tickets_total=("ID_Llamada", "count"),
    )
    tabla["dias_resolucion_prom"] = tabla["dias_resolucion_prom"].fillna(tabla["dias_resolucion_prom"].median())
    datos = tabla[config.INDICADORES_COMERCIO].copy()
    datos["tickets_mes"] = np.log1p(datos["tickets_mes"])
    datos["equipos"] = np.log1p(datos["equipos"])
    limites = datos.quantile([0.01, 0.99])
    datos = datos.clip(limites.loc[0.01], limites.loc[0.99], axis=1)
    Z = StandardScaler().fit_transform(datos)

    evaluacion = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=S).fit(Z)
        evaluacion.append({"k": k, "inercia": float(km.inertia_), "silueta": float(silhouette_score(Z, km.labels_))})

    kmeans = KMeans(n_clusters=config.K_SEGMENTOS, n_init=20, random_state=S).fit(Z)
    ward = AgglomerativeClustering(n_clusters=config.K_SEGMENTOS, linkage="ward").fit(Z)

    def nombrar(etiquetas):
        perfil = tabla[config.INDICADORES_COMERCIO].groupby(etiquetas).mean()
        if len(perfil) != 3:
            return pd.Series([f"Segmento {e}" for e in etiquetas], index=tabla.index)
        grandes = perfil["tickets_mes"].idxmax()
        hardware = perfil.drop(index=grandes)["pct_hardware"].idxmax()
        nombres = {grandes: "Grandes cuentas", hardware: "Hardware intensivo"}
        for e in perfil.index:
            nombres.setdefault(e, "Conectividad inalámbrica")
        return pd.Series(etiquetas, index=tabla.index).map(nombres)

    tabla["segmento_kmeans"] = nombrar(kmeans.labels_)
    tabla["segmento_jerarquico"] = nombrar(ward.labels_)
    pca = PCA(n_components=3, random_state=S).fit(Z)
    coordenadas = pca.transform(Z)
    for i in range(3):
        tabla[f"pc{i + 1}"] = coordenadas[:, i]
    resultado = {"evaluacion_k": evaluacion,
                 "silueta_kmeans": float(silhouette_score(Z, kmeans.labels_)),
                 "silueta_jerarquico": float(silhouette_score(Z, ward.labels_)),
                 "rand_ajustado": float(adjusted_rand_score(kmeans.labels_, ward.labels_)),
                 "varianza_pca": [float(v) for v in pca.explained_variance_ratio_]}
    return tabla.reset_index(), resultado


def principal():
    inicio = time.time()
    config.DIR_ARTEFACTOS.mkdir(exist_ok=True)
    config.DIR_DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)

    df = preparar(cargar_crudos())
    print(f"Tickets depurados: {len(df):,}")
    train, val, test = particionar(df)

    modelo_rep, m_rep = entrenar_repuesto(train, val, test)
    print(f"Repuesto (RF)      AUC prueba {m_rep['prueba']['AUC-ROC']:.3f} | F1 {m_rep['prueba']['F1']:.3f}")
    modelo_tipo, m_tipo = entrenar_tipo(train, val, test)
    print(f"Tipo de falla (RF) top-3 prueba {m_tipo['prueba']['Top-3 accuracy']:.3f}")
    modelo_reg, m_reg = entrenar_regresion(train, val, test)
    print(f"Días (GB)          MAE prueba {m_reg['prueba']['MAE']:.2f} | R2 {m_reg['prueba']['R2']:.3f}")
    modelo_dem, m_dem, serie = entrenar_demanda(df)
    print(f"Demanda (Ridge)    MAE prueba {m_dem['prueba']['MAE']:.1f} | ingenuo {m_dem['prueba_ingenuo']['MAE']:.1f}")
    segmentos, m_seg = segmentar_comercios(df)
    print(f"Segmentación       silueta K-Means {m_seg['silueta_kmeans']:.3f} | Ward {m_seg['silueta_jerarquico']:.3f}")

    joblib.dump(modelo_rep, config.DIR_ARTEFACTOS / "modelo_requiere_repuesto.joblib", compress=3)
    joblib.dump(modelo_tipo, config.DIR_ARTEFACTOS / "modelo_tipo_falla.joblib", compress=3)
    joblib.dump(modelo_reg, config.DIR_ARTEFACTOS / "modelo_dias_resolucion.joblib", compress=3)
    joblib.dump(modelo_dem, config.DIR_ARTEFACTOS / "modelo_demanda.joblib", compress=3)

    import sklearn
    metricas = {"repuesto": m_rep, "tipo_falla": m_tipo, "dias_resolucion": m_reg, "demanda": m_dem,
                "segmentacion": m_seg,
                "particion": {"entrenamiento": len(train), "validacion": len(val), "prueba": len(test)},
                "version_sklearn": sklearn.__version__, "fecha_max": str(df["CREADO_EN_dt"].max().date())}
    with open(config.DIR_ARTEFACTOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2, default=float)

    columnas = ["ID_Llamada", "CREADO_EN_dt", "CIUDAD", "CENTRO_Y_AL", "NOMBRE_EQUIPO", "TECNOLOGIA",
                "TIPO_PROBLEMA", "grupo_falla", "Tecnico", "USUARIO", "Estado", "requiere_repuesto",
                "repuesto_principal", "dias_resolucion"]
    df[columnas].to_csv(config.DIR_DATOS_PROCESADOS / "tickets_dashboard.csv.gz", index=False)
    segmentos.to_csv(config.DIR_DATOS_PROCESADOS / "segmentacion_comercios.csv", index=False)
    serie.to_csv(config.DIR_DATOS_PROCESADOS / "demanda_semanal.csv", index=False)
    print(f"Artefactos guardados en {config.DIR_ARTEFACTOS} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    principal()
