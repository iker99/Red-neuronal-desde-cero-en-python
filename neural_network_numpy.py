from pathlib import Path
import json
import numpy as np


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)  
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def mse(prediccion, real):
    return float(np.mean((prediccion - real) ** 2))


class capa:
    """Pesos W (entradas x neuronas) y sesgo b (1 x neuronas)."""

    def __init__(self, n_anterior, n_neuronas, activacion, rng):
        self.funcion_act = activacion
        # La matriz constante de 0.75 del ejemplo fuerza neuronas idénticas;
        # He inicializa distinto cada peso para aprender sin simetría.
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_anterior), (n_anterior, n_neuronas))
        self.b = np.zeros((1, n_neuronas), dtype=float)


def crear_red(semilla=42):
    rng = np.random.default_rng(semilla)
    return [capa(2, 4, "relu", rng),
            capa(4, 8, "relu", rng),
            capa(8, 1, "sigmoid", rng)]


def salida_y_cache(X, red_neuronal):
    activaciones, preactivaciones = [X], []
    for c in red_neuronal:
        z = activaciones[-1] @ c.W + c.b
        a = relu(z) if c.funcion_act == "relu" else sigmoid(z)
        preactivaciones.append(z)
        activaciones.append(a)
    return activaciones, preactivaciones


def entrenamiento(X, Y, red_neuronal, lr=0.1):
    activaciones, preactivaciones = salida_y_cache(X, red_neuronal)
    n = len(X)
    delta = (activaciones[-1] - Y) * activaciones[-1] * (1 - activaciones[-1])
    for i in reversed(range(len(red_neuronal))):
        c = red_neuronal[i]
        delta_previa = None
        if i > 0:
            delta_previa = (delta @ c.W.T) * (preactivaciones[i-1] > 0)
        c.W -= lr * activaciones[i].T @ delta / n
        c.b -= lr * np.mean(delta, axis=0, keepdims=True)
        if delta_previa is not None:
            delta = delta_previa
    return mse(activaciones[-1], Y)


def entrenar_modelo(datos, epocas=2000, tasa_aprendizaje=0.1, semilla=42):
    X = datos[["lat", "lon"]].to_numpy(dtype=float)
    Y = datos["etiqueta"].to_numpy(dtype=float).reshape(-1, 1)
    promedio = X.mean(axis=0)
    desviacion = X.std(axis=0)
    if np.any(desviacion == 0):
        raise ValueError("Las características deben tener dispersión para normalizar.")
    Xn = (X - promedio) / desviacion
    capas = crear_red(semilla)
    historial = []
    for _ in range(epocas):
        entrenamiento(Xn, Y, capas, lr=tasa_aprendizaje)
        historial.append(mse(salida_y_cache(Xn, capas)[0][-1], Y))
    modelo = {"capas": capas, "media": promedio, "desviacion": desviacion}
    return modelo, historial


def predecir(modelo, coordenadas):
    X = np.asarray(coordenadas, dtype=float)
    Xn = (X - modelo["media"]) / modelo["desviacion"]
    return salida_y_cache(Xn, modelo["capas"])[0][-1]


def guardar_modelo(modelo, destino):
    capas = modelo["capas"]
    np.savez(destino, media=modelo["media"], desviacion=modelo["desviacion"],
             W1=capas[0].W, b1=capas[0].b, W2=capas[1].W, b2=capas[1].b,
             W3=capas[2].W, b3=capas[2].b)


def cargar_modelo(ruta):
    datos = np.load(ruta, allow_pickle=False)
    red = crear_red(42)
    for i, c in enumerate(red, 1):
        c.W = datos[f"W{i}"].copy()
        c.b = datos[f"b{i}"].copy()
    return {"capas": red, "media": datos["media"], "desviacion": datos["desviacion"]}


def main():
    import pandas as pd
    from dataset import generar_puntos_gps

    destino = Path(__file__).resolve().parent
    datos = generar_puntos_gps(puntos_por_ciudad=150, semilla=42)
    modelo, historial = entrenar_modelo(datos, epocas=2000, tasa_aprendizaje=0.1, semilla=42)
    probabilidades = predecir(modelo, datos[["lat", "lon"]].to_numpy())[:, 0]
    predicciones = (probabilidades >= 0.5).astype(int)
    etiquetas = datos.etiqueta.to_numpy()
    exactitud = float(np.mean(predicciones == etiquetas))
    resultado = datos.copy()
    resultado["probabilidad_tokio"] = probabilidades
    resultado["prediccion"] = np.where(predicciones == 0, "Nueva York", "Tokio")
    resultado["acierto"] = predicciones == etiquetas
    datos.to_csv(destino / "puntos_gps_300.csv", index=False)
    resultado.to_csv(destino / "predicciones_gps_300.csv", index=False)
    guardar_modelo(modelo, destino / "pesos_gps_entrenados.npz")
    metricas = {"puntos": int(len(datos)), "por_ciudad": 150, "epocas": len(historial),
                "lr": 0.1, "arquitectura": [2, 4, 8, 1],
                "mse_inicial": historial[0], "mse_final": historial[-1],
                "exactitud_entrenamiento": exactitud,
                "aciertos": int(np.sum(predicciones == etiquetas)),
                "normalizacion_media": modelo["media"].tolist(),
                "normalizacion_desviacion": modelo["desviacion"].tolist()}
    (destino / "metricas_gps.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Puntos:", len(datos), "/ Nueva York:", sum(etiquetas == 0), "/ Tokio:", sum(etiquetas == 1))
    print("MSE inicial:", format(historial[0], ".10f"), "/ MSE final:", format(historial[-1], ".10f"))
    print("Exactitud sobre los 300 datos de entrenamiento:", f"{exactitud:.2%}")
    for i, c in enumerate(modelo["capas"], 1):
        print(f"Pesos W{i} ({c.W.shape}), sesgos b{i} ({c.b.shape})")
        print(np.array2string(c.W, precision=8))
        print(np.array2string(c.b, precision=8))
    print("Archivos generados:", "puntos_gps_300.csv, predicciones_gps_300.csv, pesos_gps_entrenados.npz, metricas_gps.json")


if __name__ == "__main__":
    main()
