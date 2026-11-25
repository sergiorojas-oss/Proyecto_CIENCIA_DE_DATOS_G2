"""Dashboard interactivo: analítica predictiva del soporte de datáfonos en campo.

Ejecución local (desde la raíz del repositorio):
    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.datos import construir_ticket, historial_comercio

st.set_page_config(page_title="Soporte de datáfonos | Analítica predictiva", page_icon="📟", layout="wide")

COLORES_SEGMENTO = {"Grandes cuentas": "#1b9e77", "Hardware intensivo": "#d95f02",
                    "Conectividad inalámbrica": "#7570b3"}
COLORES_GRUPO = {"Conectividad": "#4c78a8", "Hardware": "#e45756", "Software y transacciones": "#72b7b2",
                 "Preventivo": "#54a24b"}
ARCHIVOS = ["modelo_requiere_repuesto.joblib", "modelo_tipo_falla.joblib", "modelo_dias_resolucion.joblib",
            "modelo_demanda.joblib", "metricas.json"]
PROCESADOS = ["tickets_dashboard.csv.gz", "segmentacion_comercios.csv", "demanda_semanal.csv"]


def grafico(fig):
    """Muestra una figura de Plotly ocupando el ancho disponible en cualquier versión de Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def tabla(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------------------------
# Carga de artefactos
# --------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def asegurar_artefactos():
    faltan = [a for a in ARCHIVOS if not (config.DIR_ARTEFACTOS / a).exists()] + \
             [a for a in PROCESADOS if not (config.DIR_DATOS_PROCESADOS / a).exists()]
    if faltan:
        from src.entrenar import principal
        with st.spinner("Primera ejecución: descargando datos y entrenando los modelos (1 a 3 minutos)..."):
            principal()
    return True


@st.cache_resource(show_spinner=False)
def cargar_modelos():
    return {
        "repuesto": joblib.load(config.DIR_ARTEFACTOS / "modelo_requiere_repuesto.joblib"),
        "tipo": joblib.load(config.DIR_ARTEFACTOS / "modelo_tipo_falla.joblib"),
        "dias": joblib.load(config.DIR_ARTEFACTOS / "modelo_dias_resolucion.joblib"),
    }


@st.cache_data(show_spinner=False)
def cargar_datos():
    tickets = pd.read_csv(config.DIR_DATOS_PROCESADOS / "tickets_dashboard.csv.gz", parse_dates=["CREADO_EN_dt"])
    segmentos = pd.read_csv(config.DIR_DATOS_PROCESADOS / "segmentacion_comercios.csv")
    demanda = pd.read_csv(config.DIR_DATOS_PROCESADOS / "demanda_semanal.csv", parse_dates=["semana"])
    with open(config.DIR_ARTEFACTOS / "metricas.json", encoding="utf-8") as f:
        metricas = json.load(f)
    return tickets, segmentos, demanda, metricas


asegurar_artefactos()
modelos = cargar_modelos()
tickets, segmentos, demanda, metricas = cargar_datos()

# --------------------------------------------------------------------------------------------
# Encabezado y filtros dinámicos
# --------------------------------------------------------------------------------------------
st.title("📟 Analítica predictiva del soporte de datáfonos en campo")
st.caption("Proyecto Integrador · Maestría en Inteligencia Artificial · Universidad de La Salle · "
           "Curso CIENCIA DE DATOS_G2 · Sergio Andrés Rojas Ordóñez")

with st.sidebar:
    st.header("Filtros")
    fecha_min, fecha_max = tickets["CREADO_EN_dt"].min().date(), tickets["CREADO_EN_dt"].max().date()
    rango = st.date_input("Rango de fechas de apertura", value=(fecha_min, fecha_max),
                          min_value=fecha_min, max_value=fecha_max)
    if not isinstance(rango, (tuple, list)) or len(rango) != 2:
        rango = (fecha_min, fecha_max)

    def multiselect(etiqueta, columna):
        opciones = sorted(tickets[columna].dropna().unique())
        return st.multiselect(etiqueta, opciones, placeholder="Todas")

    f_ciudad = multiselect("Ciudad", "CIUDAD")
    f_tecnologia = multiselect("Tecnología de conectividad", "TECNOLOGIA")
    f_equipo = multiselect("Modelo de equipo", "NOMBRE_EQUIPO")
    f_grupo = multiselect("Grupo de falla", "grupo_falla")

    st.markdown("**Variable numérica**")
    max_dias = int(np.nanmax(tickets["dias_resolucion"]))
    f_dias = st.slider("Días de resolución (tickets cerrados)", 0, max_dias, (0, max_dias))
    incluir_abiertos = st.checkbox("Incluir tickets sin cierre", value=True)

filtro = tickets["CREADO_EN_dt"].between(pd.Timestamp(rango[0]), pd.Timestamp(rango[1]) + pd.Timedelta(days=1),
                                         inclusive="left")
for columna, valores in [("CIUDAD", f_ciudad), ("TECNOLOGIA", f_tecnologia), ("NOMBRE_EQUIPO", f_equipo),
                         ("grupo_falla", f_grupo)]:
    if valores:
        filtro &= tickets[columna].isin(valores)
en_rango_dias = tickets["dias_resolucion"].between(*f_dias)
filtro &= en_rango_dias | (tickets["dias_resolucion"].isna() & incluir_abiertos)
datos = tickets[filtro]

with st.sidebar:
    st.metric("Tickets seleccionados", f"{len(datos):,}", f"{len(datos) / len(tickets):.0%} del total",
              delta_color="off")

if datos.empty:
    st.warning("Ningún ticket cumple los filtros seleccionados. Amplíe el rango o quite algún filtro.")
    st.stop()

pest_resumen, pest_prediccion, pest_segmentos, pest_demanda, pest_modelos = st.tabs(
    ["📊 Resumen operativo", "🔮 Predicción en vivo", "🧩 Segmentación de comercios",
     "📈 Demanda semanal", "🧪 Desempeño de los modelos"])

# --------------------------------------------------------------------------------------------
# 1. Resumen operativo con tarjetas KPI
# --------------------------------------------------------------------------------------------
with pest_resumen:
    dias_obs = max((datos["CREADO_EN_dt"].max() - datos["CREADO_EN_dt"].min()).days + 1, 1)
    cerrados = datos["dias_resolucion"].dropna()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tickets", f"{len(datos):,}")
    c2.metric("Tickets por día", f"{len(datos) / dias_obs:,.0f}")
    c3.metric("Visitas con repuesto", f"{datos['requiere_repuesto'].mean():.1%}")
    c4.metric("Días de resolución (media)", f"{cerrados.mean():.1f}" if len(cerrados) else "–")
    c5.metric("Días de resolución (mediana)", f"{cerrados.median():.0f}" if len(cerrados) else "–")
    c6.metric("Comercios atendidos", f"{datos['USUARIO'].nunique():,}")

    st.markdown("##### Desempeño de los modelos en el conjunto de prueba")
    m = metricas
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AUC-ROC · necesidad de repuesto", f"{m['repuesto']['prueba']['AUC-ROC']:.3f}",
              f"F1 {m['repuesto']['prueba']['F1']:.3f}", delta_color="off")
    top3, ref3 = m["tipo_falla"]["prueba"]["Top-3 accuracy"], m["tipo_falla"]["prueba"]["Top-3 referencia"]
    k2.metric("Exactitud top-3 · tipo de falla", f"{top3:.1%}", f"{(top3 - ref3) * 100:+.1f} pp vs. frecuencias")
    mae, mae_d = m["dias_resolucion"]["prueba"]["MAE"], m["dias_resolucion"]["prueba_dummy"]["MAE"]
    k3.metric("MAE · días de resolución", f"{mae:.2f} días", f"{(mae / mae_d - 1):+.0%} vs. Dummy",
              delta_color="inverse")
    mae_dem, mae_ing = m["demanda"]["prueba"]["MAE"], m["demanda"]["prueba_ingenuo"]["MAE"]
    k4.metric("MAE · demanda semanal", f"{mae_dem:.1f} tickets", f"{(mae_dem / mae_ing - 1):+.0%} vs. ingenuo",
              delta_color="inverse")

    izq, der = st.columns([3, 2])
    with izq:
        semanal = (datos.groupby([pd.Grouper(key="CREADO_EN_dt", freq="W-SUN"), "grupo_falla"]).size()
                        .rename("tickets").reset_index())
        fig = px.area(semanal, x="CREADO_EN_dt", y="tickets", color="grupo_falla",
                      color_discrete_map=COLORES_GRUPO, title="Tickets por semana y grupo de falla",
                      labels={"CREADO_EN_dt": "Semana", "tickets": "Tickets", "grupo_falla": "Grupo"})
        fig.update_layout(legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
        grafico(fig)
    with der:
        por_ciudad = datos.groupby("CIUDAD").agg(tickets=("ID_Llamada", "count"),
                                                 repuesto=("requiere_repuesto", "mean")).reset_index()
        fig = px.bar(por_ciudad.sort_values("tickets"), x="tickets", y="CIUDAD", orientation="h",
                     color="repuesto", color_continuous_scale="Blues", title="Tickets por ciudad",
                     labels={"tickets": "Tickets", "CIUDAD": "", "repuesto": "% con repuesto"})
        fig.update_layout(margin=dict(t=50, b=10), coloraxis_colorbar=dict(tickformat=".0%"))
        grafico(fig)

    izq, der = st.columns(2)
    with izq:
        por_tipo = datos.groupby(["TIPO_PROBLEMA", "grupo_falla"]).agg(
            tickets=("ID_Llamada", "count"), repuesto=("requiere_repuesto", "mean"),
            dias=("dias_resolucion", "mean")).reset_index().sort_values("tickets")
        fig = px.bar(por_tipo, x="tickets", y="TIPO_PROBLEMA", color="grupo_falla", orientation="h",
                     color_discrete_map=COLORES_GRUPO, title="Tickets por tipo de falla",
                     hover_data={"repuesto": ":.1%", "dias": ":.1f"},
                     labels={"tickets": "Tickets", "TIPO_PROBLEMA": "", "grupo_falla": "Grupo",
                             "repuesto": "% con repuesto", "dias": "Días promedio"})
        fig.update_layout(margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
        grafico(fig)
    with der:
        con_cierre = datos.dropna(subset=["dias_resolucion"])
        fig = px.box(con_cierre, x="grupo_falla", y="dias_resolucion", color="grupo_falla",
                     color_discrete_map=COLORES_GRUPO, points=False, title="Días de resolución por grupo de falla",
                     labels={"grupo_falla": "", "dias_resolucion": "Días"})
        fig.update_yaxes(range=[0, np.nanpercentile(con_cierre["dias_resolucion"], 98) if len(con_cierre) else 1])
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
        grafico(fig)

    # ----------------------------------------------------------------------------------------
    # Análisis por modelo de equipo (NOMBRE_EQUIPO) y tecnología de conectividad
    # ----------------------------------------------------------------------------------------
    st.divider()
    st.markdown("##### ¿Qué equipos y tecnologías se dañan más?")
    st.caption("El número de tickets depende de cuántos terminales de cada modelo hay instalados, por eso el "
               "daño se mide con proporciones: **% de fallas de hardware** (Datáfono no enciende, Falla del pin, "
               "Daño o bloqueo, Tamper) y **% de visitas que necesitaron repuesto**. Los modelos con pocos "
               "tickets se ocultan en los porcentajes para que no distorsionen la comparación.")

    ctl1, ctl2 = st.columns(2)
    with ctl1:
        min_tickets = st.slider("Mínimo de tickets por modelo para comparar porcentajes", 10, 500, 30, 10,
                                key="min_tickets_equipo")
    with ctl2:
        top_n = st.slider("Modelos a mostrar en el volumen", 5, 30, 15, key="top_equipos")

    base = datos.assign(es_hardware=(datos["grupo_falla"] == "Hardware").astype(int))
    por_equipo = base.groupby("NOMBRE_EQUIPO").agg(
        tickets=("ID_Llamada", "count"), pct_hardware=("es_hardware", "mean"),
        pct_repuesto=("requiere_repuesto", "mean"), dias=("dias_resolucion", "mean"),
        comercios=("USUARIO", "nunique"),
        tecnologia=("TECNOLOGIA", lambda s: s.mode().iat[0] if not s.mode().empty else "Sin registro"),
    ).reset_index()
    por_equipo["tickets_por_comercio"] = por_equipo["tickets"] / por_equipo["comercios"].clip(lower=1)
    comparables = por_equipo[por_equipo["tickets"] >= min_tickets]

    izq, der = st.columns(2)
    with izq:
        top_equipos = por_equipo.nlargest(top_n, "tickets")["NOMBRE_EQUIPO"]
        vol = (base[base["NOMBRE_EQUIPO"].isin(top_equipos)]
               .groupby(["NOMBRE_EQUIPO", "grupo_falla"]).size().rename("tickets").reset_index())
        fig = px.bar(vol, x="tickets", y="NOMBRE_EQUIPO", color="grupo_falla", orientation="h",
                     color_discrete_map=COLORES_GRUPO, title=f"Tickets por modelo de equipo (top {top_n})",
                     category_orders={"NOMBRE_EQUIPO": list(top_equipos)},
                     labels={"tickets": "Tickets", "NOMBRE_EQUIPO": "", "grupo_falla": "Grupo"})
        fig.update_layout(margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2),
                          height=max(380, 26 * len(top_equipos) + 120))
        grafico(fig)
    with der:
        if comparables.empty:
            st.info("Ningún modelo alcanza el mínimo de tickets con los filtros actuales; baje el mínimo.")
        else:
            fig = px.scatter(comparables, x="pct_hardware", y="pct_repuesto", size="tickets", color="tecnologia",
                             hover_name="NOMBRE_EQUIPO", size_max=40,
                             hover_data={"tickets": ":,", "dias": ":.1f", "comercios": ":,",
                                         "pct_hardware": ":.1%", "pct_repuesto": ":.1%", "tecnologia": True},
                             title="Mapa de daño por modelo (arriba a la derecha = se daña más)",
                             labels={"pct_hardware": "% fallas de hardware", "pct_repuesto": "% con repuesto",
                                     "tecnologia": "Tecnología principal", "tickets": "Tickets",
                                     "dias": "Días promedio", "comercios": "Comercios"})
            fig.add_vline(x=base["es_hardware"].mean(), line_dash="dot", line_color="grey")
            fig.add_hline(y=base["requiere_repuesto"].mean(), line_dash="dot", line_color="grey")
            fig.update_layout(margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2), height=460,
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            grafico(fig)
            st.caption("Las líneas punteadas son el promedio de los tickets filtrados. Los modelos del cuadrante "
                       "superior derecho fallan por hardware y consumen repuestos por encima del promedio.")

    izq, der = st.columns(2)
    with izq:
        por_tec = base.groupby("TECNOLOGIA").agg(
            tickets=("ID_Llamada", "count"), pct_hardware=("es_hardware", "mean"),
            pct_repuesto=("requiere_repuesto", "mean"), dias=("dias_resolucion", "mean")).reset_index()
        mezcla = (base.groupby(["TECNOLOGIA", "grupo_falla"]).size().rename("tickets").reset_index())
        mezcla["participacion"] = mezcla["tickets"] / mezcla.groupby("TECNOLOGIA")["tickets"].transform("sum")
        orden_tec = por_tec.sort_values("pct_hardware", ascending=False)["TECNOLOGIA"].tolist()
        fig = px.bar(mezcla, x="TECNOLOGIA", y="participacion", color="grupo_falla",
                     color_discrete_map=COLORES_GRUPO, category_orders={"TECNOLOGIA": orden_tec},
                     hover_data={"tickets": ":,", "participacion": ":.1%"},
                     title="Tipo de falla dentro de cada tecnología",
                     labels={"TECNOLOGIA": "", "participacion": "% de los tickets", "grupo_falla": "Grupo",
                             "tickets": "Tickets"})
        fig.update_layout(margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2), yaxis_tickformat=".0%")
        grafico(fig)
    with der:
        metrica_mapa = st.radio("Métrica del mapa de calor", ["% fallas de hardware", "% con repuesto", "Tickets"],
                                horizontal=True, key="metrica_mapa_equipo")
        columna_mapa = {"% fallas de hardware": "es_hardware", "% con repuesto": "requiere_repuesto",
                        "Tickets": "ID_Llamada"}[metrica_mapa]
        agregacion = "count" if metrica_mapa == "Tickets" else "mean"
        cruce = base[base["NOMBRE_EQUIPO"].isin(comparables["NOMBRE_EQUIPO"])]
        if cruce.empty:
            st.info("Sin modelos suficientes para el mapa de calor con los filtros actuales.")
        else:
            matriz = cruce.pivot_table(index="NOMBRE_EQUIPO", columns="TECNOLOGIA", values=columna_mapa,
                                       aggfunc=agregacion)
            conteo = cruce.pivot_table(index="NOMBRE_EQUIPO", columns="TECNOLOGIA", values="ID_Llamada",
                                       aggfunc="count")
            if agregacion == "mean":
                matriz = matriz.where(conteo >= 10)  # celdas con menos de 10 tickets no son confiables
            matriz = matriz.loc[matriz.max(axis=1).sort_values(ascending=False).index]
            fig = px.imshow(matriz, aspect="auto", color_continuous_scale="Reds",
                            text_auto=",.0f" if agregacion == "count" else ".0%",
                            title=f"Modelo × tecnología · {metrica_mapa}",
                            labels={"x": "", "y": "", "color": metrica_mapa})
            if agregacion == "mean":
                fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
            fig.update_layout(margin=dict(t=50, b=10), height=max(380, 26 * len(matriz) + 120))
            grafico(fig)
            if agregacion == "mean":
                st.caption("Las celdas vacías tienen menos de 10 tickets o no existen en los datos.")

    with st.expander("Ranking de modelos de equipo por daño", expanded=False):
        ranking = comparables.sort_values(["pct_hardware", "pct_repuesto"], ascending=False)
        ranking = ranking.assign(pct_hardware=ranking["pct_hardware"] * 100,
                                 pct_repuesto=ranking["pct_repuesto"] * 100)
        tabla(ranking[["NOMBRE_EQUIPO", "tecnologia", "tickets", "comercios", "tickets_por_comercio",
                       "pct_hardware", "pct_repuesto", "dias"]]
              .rename(columns={"NOMBRE_EQUIPO": "Modelo", "tecnologia": "Tecnología principal",
                               "tickets": "Tickets", "comercios": "Comercios",
                               "tickets_por_comercio": "Tickets por comercio", "pct_hardware": "% hardware",
                               "pct_repuesto": "% con repuesto", "dias": "Días promedio"}).round(1),
              hide_index=True)

    repuestos = (datos.loc[datos["repuesto_principal"] != "Ninguno", "repuesto_principal"]
                 .value_counts().head(8).rename_axis("repuesto").reset_index(name="visitas"))
    if not repuestos.empty:
        fig = px.bar(repuestos, x="repuesto", y="visitas", title="Repuestos más usados (repuesto principal)",
                     labels={"repuesto": "", "visitas": "Visitas"}, color_discrete_sequence=["#4c78a8"])
        fig.update_layout(margin=dict(t=50, b=10), height=320)
        grafico(fig)

# --------------------------------------------------------------------------------------------
# 2. Predicción en vivo
# --------------------------------------------------------------------------------------------
with pest_prediccion:
    st.subheader("Simulador de despacho para un ticket nuevo")
    st.caption("Ingrese los datos disponibles al abrir el ticket. Los modelos solo usan información conocida "
               "en ese momento (auditoría de fuga de datos del cuaderno, sección 5.3).")

    ciudades = sorted(tickets["CIUDAD"].unique())
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ciudad = st.selectbox("Ciudad", ciudades)
        centros = sorted(tickets.loc[tickets["CIUDAD"] == ciudad, "CENTRO_Y_AL"].unique())
        centro = st.selectbox("Centro de distribución", centros)
        tecnicos = sorted(tickets.loc[tickets["CIUDAD"] == ciudad, "Tecnico"].dropna().unique())
        tecnico = st.selectbox("Técnico asignado", tecnicos)
    with col_b:
        equipo = st.selectbox("Modelo de equipo", sorted(tickets["NOMBRE_EQUIPO"].unique()))
        tecnologias_equipo = tickets.loc[tickets["NOMBRE_EQUIPO"] == equipo, "TECNOLOGIA"].value_counts().index
        tecnologia = st.selectbox("Tecnología", list(tecnologias_equipo) +
                                  [t for t in sorted(tickets["TECNOLOGIA"].unique()) if t not in tecnologias_equipo])
        tipo_reportado = st.selectbox("Falla reportada por el comercio", sorted(tickets["TIPO_PROBLEMA"].unique()))
    with col_c:
        fecha = st.date_input("Fecha de apertura",
                              value=(tickets["CREADO_EN_dt"].max() + pd.Timedelta(days=1)).date())
        comercios_ciudad = (tickets.loc[tickets["CIUDAD"] == ciudad, "USUARIO"].value_counts().index.tolist())
        comercio = st.selectbox("Comercio", ["Comercio nuevo"] + comercios_ciudad,
                                help="El historial del comercio se calcula con sus tickets anteriores a la fecha.")
        umbral = st.slider("Umbral de despacho de repuesto", 0.2, 0.8, 0.3, 0.05,
                           help="0.3 maximiza la cobertura; valores mayores reducen el inventario transportado.")

    if comercio == "Comercio nuevo":
        previos, dias_ultimo = 0, -1
    else:
        previos, dias_ultimo = historial_comercio(tickets, comercio, fecha)

    ticket = construir_ticket(equipo, tecnologia, ciudad, centro, tipo_reportado, tecnico, fecha,
                              previos, dias_ultimo)
    p_rep = float(modelos["repuesto"].predict_proba(ticket[config.ENTRADAS_REPUESTO])[:, 1][0])
    dias_est = float(modelos["dias"].predict(ticket[config.ENTRADAS_REGRESION])[0])
    prob_tipo = modelos["tipo"].predict_proba(ticket[config.ENTRADAS_TIPO])[0]
    kit = (pd.DataFrame({"falla": modelos["tipo"].classes_, "probabilidad": prob_tipo})
             .sort_values("probabilidad", ascending=False).head(3))

    st.divider()
    r1, r2, r3 = st.columns([1.1, 1, 1.2])
    with r1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p_rep * 100, number={"suffix": " %"},
            title={"text": "Probabilidad de requerir repuesto"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4c78a8"},
                   "threshold": {"line": {"color": "#e45756", "width": 4}, "value": umbral * 100},
                   "steps": [{"range": [0, umbral * 100], "color": "#eef2f7"},
                             {"range": [umbral * 100, 100], "color": "#dbe6f3"}]}))
        fig.update_layout(height=260, margin=dict(t=60, b=0, l=20, r=20))
        grafico(fig)
        if p_rep >= umbral:
            st.success(f"**Despachar con repuesto** (probabilidad {p_rep:.0%} ≥ umbral {umbral:.0%})")
        else:
            st.info(f"**Despachar sin repuesto** (probabilidad {p_rep:.0%} < umbral {umbral:.0%})")
    with r2:
        st.metric("Días estimados de resolución", f"{dias_est:.1f} días",
                  help="Gradient Boosting; error absoluto medio en prueba "
                       f"{metricas['dias_resolucion']['prueba']['MAE']:.2f} días.")
        st.metric("Historial del comercio", f"{previos:,} tickets previos",
                  "primer ticket" if dias_ultimo < 0 else f"último hace {dias_ultimo} día{'' if dias_ultimo == 1 else 's'}", delta_color="off")
        sugeridos = (tickets.loc[(tickets["TIPO_PROBLEMA"] == tipo_reportado) & (tickets["requiere_repuesto"] == 1),
                                 "repuesto_principal"].value_counts(normalize=True).head(3))
        if p_rep >= umbral and not sugeridos.empty:
            st.markdown("**Repuestos a cargar** (frecuencia histórica para esta falla)")
            for nombre, frecuencia in sugeridos.items():
                st.markdown(f"- {nombre}: {frecuencia:.0%}")
    with r3:
        fig = px.bar(kit.iloc[::-1], x="probabilidad", y="falla", orientation="h",
                     title="Kit sugerido: 3 fallas más probables según el equipo",
                     labels={"probabilidad": "Probabilidad", "falla": ""}, text_auto=".0%",
                     color_discrete_sequence=["#72b7b2"])
        fig.update_layout(height=260, margin=dict(t=60, b=0), xaxis_tickformat=".0%")
        grafico(fig)
        if tipo_reportado in kit["falla"].values:
            st.caption("La falla reportada está dentro del kit sugerido.")

    with st.expander("Política de despacho: cobertura frente a uso del inventario (conjunto de prueba)"):
        politica = pd.DataFrame(metricas["repuesto"]["politica"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=politica["umbral"], y=politica["cobertura_recall"], mode="lines+markers",
                                 name="Cobertura (recall)"))
        fig.add_trace(go.Scatter(x=politica["umbral"], y=politica["uso_precision"], mode="lines+markers",
                                 name="Uso de repuestos (precisión)"))
        fig.add_vline(x=umbral, line_dash="dash", line_color="#e45756")
        fig.update_layout(xaxis_title="Umbral de probabilidad", yaxis_title="%", height=330,
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
        grafico(fig)
        tabla(politica.rename(columns={"umbral": "Umbral", "pct_despachadas_con_repuesto": "% despachadas con repuesto",
                                       "cobertura_recall": "Cobertura %", "uso_precision": "Uso del repuesto %",
                                       "visitas_sin_cubrir": "Visitas sin cubrir"}).round(1), hide_index=True)
        st.caption(f"Sin modelo, llevar repuesto siempre implica que solo el "
                   f"{metricas['repuesto']['tasa_base']:.1%} de los repuestos transportados se usa.")

# --------------------------------------------------------------------------------------------
# 3. Segmentación de comercios (mapa de clústeres)
# --------------------------------------------------------------------------------------------
with pest_segmentos:
    seg = segmentos.copy()
    if f_ciudad:
        seg = seg[seg["ciudad_principal"].isin(f_ciudad)]
    c_izq, c_der = st.columns([1, 3])
    with c_izq:
        algoritmo = st.radio("Algoritmo", ["K-Means", "Jerárquico (Ward)"])
        vista = st.radio("Vista", ["2D", "3D"], horizontal=True)
        columna_seg = "segmento_kmeans" if algoritmo == "K-Means" else "segmento_jerarquico"
        ms = metricas["segmentacion"]
        st.metric("Silueta", f"{ms['silueta_kmeans' if algoritmo == 'K-Means' else 'silueta_jerarquico']:.3f}")
        st.metric("Rand ajustado entre algoritmos", f"{ms['rand_ajustado']:.2f}")
        st.metric("Comercios mostrados", f"{len(seg):,}")
    with c_der:
        var = ms["varianza_pca"]
        hover = {"USUARIO": True, "ciudad_principal": True, "tickets_mes": ":.1f", "pct_hardware": ":.0f",
                 "pct_inalambrico": ":.0f", "pc1": False, "pc2": False}
        if vista == "2D":
            fig = px.scatter(seg, x="pc1", y="pc2", color=columna_seg, color_discrete_map=COLORES_SEGMENTO,
                             size=np.sqrt(seg["tickets_mes"].clip(lower=0.1)), size_max=18,
                             hover_data=hover, category_orders={columna_seg: config.ORDEN_SEGMENTOS},
                             labels={"pc1": f"Componente 1 ({var[0]:.0%})", "pc2": f"Componente 2 ({var[1]:.0%})",
                                     columna_seg: "Segmento"},
                             title=f"Comercios en el espacio PCA · {algoritmo}")
        else:
            fig = px.scatter_3d(seg, x="pc1", y="pc2", z="pc3", color=columna_seg,
                                color_discrete_map=COLORES_SEGMENTO, hover_data=hover,
                                category_orders={columna_seg: config.ORDEN_SEGMENTOS},
                                labels={"pc1": "PC1", "pc2": "PC2", "pc3": "PC3", columna_seg: "Segmento"},
                                title=f"Comercios en el espacio PCA (3D) · {algoritmo}")
            fig.update_traces(marker=dict(size=3))
        fig.update_layout(height=480, legend=dict(orientation="h", y=-0.12), margin=dict(t=50))
        grafico(fig)

    perfil = seg.groupby(columna_seg).agg(
        comercios=("USUARIO", "count"), **{c: (c, "mean") for c in config.INDICADORES_COMERCIO})
    perfil["% de los tickets"] = seg.groupby(columna_seg)["tickets_total"].sum() / seg["tickets_total"].sum() * 100
    perfil = perfil.reindex([s for s in config.ORDEN_SEGMENTOS if s in perfil.index])
    st.markdown("##### Perfil promedio de cada segmento")
    tabla(perfil.rename(columns={
        "comercios": "Comercios", "tickets_mes": "Tickets/mes", "equipos": "Equipos",
        "pct_conectividad": "% conectividad", "pct_hardware": "% hardware", "pct_preventivo": "% preventivo",
        "pct_requiere_repuesto": "% con repuesto", "dias_resolucion_prom": "Días de resolución",
        "pct_inalambrico": "% inalámbricos"}).round(1))

    cols = st.columns(len(perfil))
    for col, (nombre, fila) in zip(cols, perfil.iterrows()):
        with col:
            st.markdown(f"**{nombre}** · {int(fila['comercios'])} comercios, {fila['% de los tickets']:.0f} % de los tickets")
            st.caption(config.ESTRATEGIAS_SEGMENTO.get(nombre, ""))

    with st.expander("Selección del número de clústeres: método del codo y coeficiente de silueta"):
        ev = pd.DataFrame(ms["evaluacion_k"])
        e1, e2 = st.columns(2)
        with e1:
            fig = px.line(ev, x="k", y="inercia", markers=True, title="Método del codo",
                          labels={"inercia": "Inercia intra-clúster"})
            grafico(fig)
        with e2:
            fig = px.line(ev, x="k", y="silueta", markers=True, title="Coeficiente de silueta",
                          labels={"silueta": "Silueta"})
            grafico(fig)

    buscado = st.selectbox("Consultar un comercio", ["—"] + sorted(seg["USUARIO"]))
    if buscado != "—":
        fila = seg[seg["USUARIO"] == buscado].iloc[0]
        st.info(f"**{buscado}** ({fila['ciudad_principal']}) pertenece al segmento **{fila[columna_seg]}**: "
                f"{fila['tickets_mes']:.1f} tickets/mes, {fila['equipos']:.0f} equipos, "
                f"{fila['pct_hardware']:.0f} % de fallas de hardware y {fila['pct_inalambrico']:.0f} % de terminales "
                f"inalámbricos. Estrategia: {config.ESTRATEGIAS_SEGMENTO.get(fila[columna_seg], '')}")

# --------------------------------------------------------------------------------------------
# 4. Demanda semanal
# --------------------------------------------------------------------------------------------
with pest_demanda:
    ciudades_dem = f_ciudad or sorted(demanda["CIUDAD"].unique())
    dem = demanda[demanda["CIUDAD"].isin(ciudades_dem)]
    proxima = dem[dem["tipo"] == "Próxima semana"]
    if not proxima.empty:
        st.markdown(f"##### Pronóstico para la semana que termina el {proxima['semana'].max():%d/%m/%Y}")
        cols = st.columns(min(len(proxima), 6))
        for col, (_, fila) in zip(cols, proxima.iterrows()):
            ultima = dem[(dem["CIUDAD"] == fila["CIUDAD"]) & dem["tickets"].notna()].sort_values("semana")
            previo = ultima["tickets"].iloc[-1] if len(ultima) else np.nan
            col.metric(fila["CIUDAD"].title(), f"{fila['pronostico']:.0f} tickets",
                       f"{fila['pronostico'] - previo:+.0f} vs. última semana" if pd.notna(previo) else None)

    largo = dem.melt(id_vars=["semana", "CIUDAD", "tipo"], value_vars=["tickets", "pronostico"],
                     var_name="serie", value_name="valor").dropna(subset=["valor"])
    largo["serie"] = largo["serie"].map({"tickets": "Real", "pronostico": "Pronóstico"})
    fig = px.line(largo, x="semana", y="valor", color="CIUDAD", line_dash="serie",
                  title="Tickets por semana: real y pronóstico (Ridge con rezagos)",
                  labels={"semana": "Semana", "valor": "Tickets", "CIUDAD": "Ciudad", "serie": ""})
    corte = metricas["demanda"]["corte_prueba"]
    fig.add_vline(x=corte, line_dash="dot", line_color="grey")
    fig.add_annotation(x=corte, y=1, yref="paper", text="inicio de prueba", showarrow=False, xanchor="left")
    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.2))
    grafico(fig)
    st.caption(f"Error absoluto medio en prueba: {metricas['demanda']['prueba']['MAE']:.1f} tickets por semana "
               f"y ciudad, frente a {metricas['demanda']['prueba_ingenuo']['MAE']:.1f} del pronóstico ingenuo.")

# --------------------------------------------------------------------------------------------
# 5. Desempeño de los modelos
# --------------------------------------------------------------------------------------------
with pest_modelos:
    st.markdown("##### Necesidad de repuesto (validación)")
    tabla(pd.DataFrame(metricas["repuesto"]["validacion"]).T.round(3))
    st.caption("En el cuaderno, el perceptrón multicapa en PyTorch obtuvo un AUC-ROC de 0.844 en validación y "
               "0.841 en prueba, el mismo nivel de la regresión logística y del Random Forest.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Tipo de falla (Random Forest)")
        tabla(pd.DataFrame(metricas["tipo_falla"]).round(3))
    with c2:
        st.markdown("##### Días de resolución (Gradient Boosting)")
        r = metricas["dias_resolucion"]
        tabla(pd.DataFrame({"Gradient Boosting (validación)": r["validacion"], "Gradient Boosting (prueba)": r["prueba"],
                            "Dummy (prueba)": r["prueba_dummy"]}).T.round(3))

    resumen = pd.DataFrame([
        ("Necesidad de repuesto", "Random Forest", "AUC-ROC", m["repuesto"]["prueba"]["AUC-ROC"], "≥ 0.80"),
        ("Necesidad de repuesto", "Random Forest", "F1", m["repuesto"]["prueba"]["F1"], "≥ 0.75"),
        ("Tipo de falla", "Random Forest", "Exactitud top-3", top3, f"≥ {ref3 + 0.15:.3f}"),
        ("Días de resolución", "Gradient Boosting", "MAE (días)", mae, f"≤ {0.85 * mae_d:.2f}"),
        ("Demanda semanal", "Ridge", "MAE (tickets)", mae_dem, f"< {mae_ing:.1f}"),
        ("Segmentación", "K-Means", "Silueta", metricas["segmentacion"]["silueta_kmeans"], "≥ 0.25"),
    ], columns=["Tarea", "Modelo", "Métrica", "Prueba", "Meta del objetivo"])
    st.markdown("##### Evaluación final frente a las metas de los objetivos específicos")
    tabla(resumen.round(3), hide_index=True)
    st.caption(f"Partición estratificada 70/15/15: {metricas['particion']['entrenamiento']:,} / "
               f"{metricas['particion']['validacion']:,} / {metricas['particion']['prueba']:,} tickets. "
               f"Modelos entrenados con scikit-learn {metricas['version_sklearn']}.")
