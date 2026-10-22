import numpy as np
import pandas as pd

CENTROS = {
    "Nueva York": {"lat": 40.7128, "lon": -74.0060, "lat_sd": 0.018, "lon_sd": 0.025},
    "Tokio": {"lat": 35.6762, "lon": 139.6503, "lat_sd": 0.025, "lon_sd": 0.035},
}


def generar_puntos_gps(puntos_por_ciudad=150, semilla=42):
    """Devuelve DataFrame reproducible: id, ciudad, lat, lon, etiqueta.

    0=Nueva York, 1=Tokio. Latitud y longitud en grados decimales.
    """
    if puntos_por_ciudad < 1:
        raise ValueError("puntos_por_ciudad debe ser positivo")
    rng = np.random.default_rng(semilla)
    filas = []
    for clase, (ciudad, config) in enumerate(CENTROS.items()):
        latitudes = rng.normal(config["lat"], config["lat_sd"], puntos_por_ciudad)
        longitudes = rng.normal(config["lon"], config["lon_sd"], puntos_por_ciudad)
        for lat, lon in zip(latitudes, longitudes):
            filas.append((len(filas) + 1, ciudad, round(float(lat), 6), round(float(lon), 6), clase))
    return pd.DataFrame(filas, columns=["id", "ciudad", "lat", "lon", "etiqueta"])


def main():
    import streamlit as st
    from neural_network_numpy import entrenar_modelo, predecir

    st.set_page_config(page_title="GPS: Nueva York y Tokio", page_icon="🌐", layout="wide")
    st.title("Clasificación GPS: Nueva York y Tokio")
    st.caption("300 coordenadas sintéticas reproducibles (150 por ciudad), semilla 42. No son observaciones GPS reales.")

    df = generar_puntos_gps(150, 42)
    a, b, c = st.columns(3)
    a.metric("Puntos GPS", len(df))
    b.metric("Nueva York", int((df.etiqueta == 0).sum()))
    c.metric("Tokio", int((df.etiqueta == 1).sum()))

    tab1, tab2 = st.tabs(["🗽 Mapa de Nueva York", "🗼 Mapa de Tokio"])
    with tab1:
        st.map(df.loc[df.ciudad == "Nueva York", ["lat", "lon"]], zoom=10)
    with tab2:
        st.map(df.loc[df.ciudad == "Tokio", ["lat", "lon"]], zoom=10)

    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Descargar 300 puntos (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="puntos_gps_300.csv", mime="text/csv")

    st.subheader("Red neuronal NumPy (2 → 4 → 8 → 1)")
    st.write("Entrena con las 300 coordenadas normalizadas y muestra las probabilidades calculadas por la red.")
    if st.button("Entrenar red neuronal", type="primary"):
        modelo, historial = entrenar_modelo(df, epocas=2000, tasa_aprendizaje=0.1, semilla=42)
        probabilidades = predecir(modelo, df[["lat", "lon"]].to_numpy())[:, 0]
        etiquetas_predichas = (probabilidades >= 0.5).astype(int)
        exactitud = float(np.mean(etiquetas_predichas == df.etiqueta.to_numpy()))
        st.metric("Exactitud sobre los 300 datos de entrenamiento", f"{exactitud:.2%}")
        st.metric("Error cuadrático medio (MSE)", f"{historial[-1]:.8f}")
        st.line_chart(pd.DataFrame({"MSE": historial}, index=np.arange(1, len(historial) + 1)))
        resultado = df.copy()
        resultado["probabilidad_tokio"] = probabilidades
        resultado["prediccion"] = np.where(etiquetas_predichas == 0, "Nueva York", "Tokio")
        st.dataframe(resultado, use_container_width=True, hide_index=True)
        st.info("La exactitud mostrada es de entrenamiento; el Excel verifica numéricamente pesos y cálculos, no generalización a datos nuevos.")


if __name__ == "__main__":
    main()
