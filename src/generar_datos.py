"""
Generador del dataset semisintético de Automuelles Virtual.

No existen datos históricos reales, así que los casos se generan a partir de
reglas técnicas de suspensión (qué síntomas produce cada falla y en qué
vehículos es más frecuente) más ruido aleatorio. Todas las suposiciones están
en las constantes de este archivo para que se puedan revisar y ajustar con un
técnico.

Uso:  python src/generar_datos.py  ->  data/dataset_automuelles.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEMILLA = 42
N_CASOS = 5000
rng = np.random.default_rng(SEMILLA)

# ---------------------------------------------------------------------------
# 1. Vehículos. Las proporciones imitan un taller de suspensión colombiano:
#    muchos particulares y pocos vehículos de carga (sesgo de representación).
# ---------------------------------------------------------------------------
TIPOS = {
    "particular":       0.55,
    "camioneta":        0.22,
    "servicio_publico": 0.13,
    "carga":            0.10,
}

# (marca, origen, peso relativo dentro del tipo)
MARCAS = {
    "particular": [("Chevrolet", "tradicional", 22), ("Renault", "tradicional", 18),
                   ("Mazda", "tradicional", 14), ("Kia", "tradicional", 12),
                   ("Toyota", "tradicional", 10), ("Nissan", "tradicional", 8),
                   ("Hyundai", "tradicional", 7), ("Suzuki", "tradicional", 5),
                   ("BYD", "china", 2), ("Chery", "china", 1), ("JAC", "china", 1)],
    "camioneta":  [("Toyota", "tradicional", 30), ("Chevrolet", "tradicional", 20),
                   ("Nissan", "tradicional", 15), ("Mazda", "tradicional", 10),
                   ("Ford", "tradicional", 8), ("Mitsubishi", "tradicional", 7),
                   ("JAC", "china", 5), ("Great Wall", "china", 3), ("Foton", "china", 2)],
    "servicio_publico": [("Hyundai", "tradicional", 35), ("Kia", "tradicional", 25),
                         ("Chevrolet", "tradicional", 20), ("Renault", "tradicional", 10),
                         ("JAC", "china", 6), ("BYD", "china", 4)],
    "carga":      [("Chevrolet", "tradicional", 35), ("Hino", "tradicional", 20),
                   ("Isuzu", "tradicional", 15), ("International", "tradicional", 5),
                   ("Foton", "china", 15), ("JAC", "china", 10)],
}

CLASES = ["amortiguador_danado", "muelle_fatigado", "espiral_vencido",
          "bujes_desgastados", "terminal_rotula", "sin_falla_suspension"]

# Probabilidad base de cada falla según el tipo de vehículo.
# Carga y camioneta usan muelles (ballestas); particulares y taxis, espirales.
PRIOR_POR_TIPO = {
    "particular":       [0.27, 0.01, 0.20, 0.20, 0.17, 0.15],
    "camioneta":        [0.24, 0.20, 0.08, 0.18, 0.17, 0.13],
    "servicio_publico": [0.28, 0.02, 0.22, 0.22, 0.18, 0.08],
    "carga":            [0.18, 0.42, 0.01, 0.17, 0.14, 0.08],
}

# ---------------------------------------------------------------------------
# 2. Síntomas: probabilidad de que cada falla produzca cada síntoma.
# ---------------------------------------------------------------------------
SINTOMAS = ["rebote_excesivo", "golpe_en_huecos", "inclinacion_lateral",
            "vehiculo_bajo", "desgaste_irregular_llantas", "vibracion_volante",
            "juego_en_direccion", "ruido_al_girar", "fuga_aceite"]

PERFIL_SINTOMAS = {  # orden de SINTOMAS
    "amortiguador_danado":  [0.80, 0.45, 0.35, 0.10, 0.50, 0.20, 0.05, 0.10, 0.50],
    "muelle_fatigado":      [0.25, 0.60, 0.55, 0.75, 0.30, 0.10, 0.05, 0.10, 0.03],
    "espiral_vencido":      [0.25, 0.55, 0.50, 0.70, 0.25, 0.10, 0.05, 0.25, 0.03],
    "bujes_desgastados":    [0.15, 0.80, 0.15, 0.10, 0.35, 0.40, 0.30, 0.35, 0.03],
    "terminal_rotula":      [0.05, 0.40, 0.10, 0.05, 0.60, 0.60, 0.75, 0.50, 0.02],
    "sin_falla_suspension": [0.08, 0.12, 0.08, 0.05, 0.15, 0.30, 0.05, 0.08, 0.02],
}

# ---------------------------------------------------------------------------
# 3. Frases en lenguaje coloquial para la descripción libre del cliente.
# ---------------------------------------------------------------------------
FRASES = {
    "rebote_excesivo": ["el carro rebota mucho después de un policía acostado",
                        "queda saltando cuando paso un resalto", "se siente como un barco, rebota",
                        "brinca demasiado en la carretera"],
    "golpe_en_huecos": ["suena un golpe cuando paso por un hueco", "traquea adelante en las calles destapadas",
                        "se escucha un clac en los huecos", "golpea seco en los baches"],
    "inclinacion_lateral": ["está ladeado hacia un lado", "se va de lado en las curvas",
                            "se ve más bajito de un lado", "se recuesta mucho al girar"],
    "vehiculo_bajo": ["la parte de atrás está caída", "se baja mucho cuando lo cargo",
                      "quedó muy bajito", "toca el piso cuando va cargado"],
    "desgaste_irregular_llantas": ["las llantas se gastan por dentro", "las llantas tienen desgaste disparejo",
                                   "se comió las llantas por un lado"],
    "vibracion_volante": ["el timón vibra a más de 80", "tiembla la cabrilla en carretera",
                          "vibra el volante"],
    "juego_en_direccion": ["se siente flojo el timón", "la dirección tiene juego",
                           "se va de lado solo, toca corregirlo"],
    "ruido_al_girar": ["cruje cuando giro", "suena al voltear el timón", "chilla al hacer giros"],
    "fuga_aceite": ["sale aceite del amortiguador", "el amortiguador está sudando aceite",
                    "tiene una mancha de aceite en la llanta"],
}
RELLENO = ["buenas tardes", "desde hace un mes", "hace poco", "por favor ayuda", "", "", ""]


def frase_con_errores(texto: str) -> str:
    """Simula errores de digitación: borra una letra el 8 % de las veces."""
    if texto and rng.random() < 0.08:
        i = rng.integers(0, len(texto))
        texto = texto[:i] + texto[i + 1:]
    return texto


def describir(sintomas_activos: list[str]) -> str:
    """Arma la descripción libre que escribiría el cliente."""
    if rng.random() < 0.25:          # una cuarta parte de los clientes no escribe nada
        return ""
    elegidos = list(sintomas_activos)
    rng.shuffle(elegidos)
    elegidos = elegidos[: rng.integers(1, 4)]
    if rng.random() < 0.15:          # a veces mencionan algo que no marcaron
        elegidos.append(rng.choice(SINTOMAS))
    partes = [rng.choice(RELLENO)] + [rng.choice(FRASES[s]) for s in elegidos]
    return frase_con_errores(", ".join(p for p in partes if p))


def generar_caso() -> dict:
    tipo = rng.choice(list(TIPOS), p=list(TIPOS.values()))
    opciones = MARCAS[tipo]
    pesos = np.array([p for _, _, p in opciones], dtype=float)
    marca, origen, _ = opciones[rng.choice(len(opciones), p=pesos / pesos.sum())]

    anio = int(rng.integers(2000, 2026))
    edad = 2026 - anio
    km_por_anio = {"particular": 12_000, "camioneta": 18_000,
                   "servicio_publico": 60_000, "carga": 45_000}[tipo]
    kilometraje = int(max(1_000, rng.normal(km_por_anio * max(edad, 0.5), km_por_anio * 2)))
    carga = rng.choice(["baja", "media", "alta"],
                       p=[0.1, 0.3, 0.6] if tipo == "carga" else [0.5, 0.35, 0.15])
    via = rng.choice(["urbana", "mixta", "destapada"], p=[0.5, 0.35, 0.15])

    # Probabilidad de cada falla: prior del tipo, ajustado por uso del vehículo
    p = np.array(PRIOR_POR_TIPO[tipo], dtype=float)
    if kilometraje > 150_000: p[[0, 3]] *= 1.4          # más km: amortiguadores y bujes
    if via == "destapada":    p[[3, 4]] *= 1.5          # destapado: bujes y terminales
    if carga == "alta":       p[[1, 2]] *= 1.6          # sobrecarga: muelles y espirales
    if edad <= 2:             p[5] *= 2.5               # vehículo nuevo: suele no tener falla
    falla = rng.choice(CLASES, p=p / p.sum())

    activos = {s: int(rng.random() < prob) for s, prob in zip(SINTOMAS, PERFIL_SINTOMAS[falla])}

    # Ruido de etiqueta: diagnósticos ambiguos o mal registrados.
    # En vehículos de carga es mayor (suposición: varias fallas simultáneas por sobrecarga).
    ruido = 0.12 if tipo == "carga" else 0.06
    if rng.random() < ruido:
        falla = rng.choice(CLASES)

    return {
        "tipo_vehiculo": tipo, "marca": marca, "origen_marca": origen,
        "anio": anio, "kilometraje": kilometraje,
        "carga_habitual": carga, "tipo_via": via,
        **activos,
        "descripcion": describir([s for s, v in activos.items() if v]),
        "falla": falla,
    }


def main():
    df = pd.DataFrame([generar_caso() for _ in range(N_CASOS)])
    salida = Path(__file__).resolve().parents[1] / "data" / "dataset_automuelles.csv"
    salida.parent.mkdir(exist_ok=True)
    df.to_csv(salida, index=False, encoding="utf-8")
    print(f"Dataset generado: {salida} ({len(df)} casos)")
    print(df["falla"].value_counts().to_string())
    print(df["tipo_vehiculo"].value_counts().to_string())


if __name__ == "__main__":
    main()
