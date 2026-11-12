"""
Entrenamiento y evaluación del modelo de diagnóstico de Automuelles Virtual.

1. División estratificada 70 / 15 / 15 (entrenamiento, validación, prueba).
2. Preprocesamiento: one-hot (categóricas), escalado (numéricas), TF-IDF (texto libre).
3. Comparación de modelos: regresión logística (línea base), Random Forest y LightGBM.
4. Ajuste de hiperparámetros de LightGBM con Optuna sobre el conjunto de validación.
5. Evaluación final en prueba: accuracy, F1-macro, top-3 accuracy, matriz de confusión.
6. Equidad con fairlearn: desempeño por tipo de vehículo, origen de marca y antigüedad,
   y mitigación por re-ponderación de muestras.

Uso:  python src/entrenar.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from fairlearn.metrics import MetricFrame
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEMILLA = 42
RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "reports" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
(RAIZ / "models").mkdir(exist_ok=True)

CATEGORICAS = ["tipo_vehiculo", "marca", "origen_marca", "carga_habitual", "tipo_via"]
NUMERICAS = ["anio", "kilometraje"]
SINTOMAS = ["rebote_excesivo", "golpe_en_huecos", "inclinacion_lateral", "vehiculo_bajo",
            "desgaste_irregular_llantas", "vibracion_volante", "juego_en_direccion",
            "ruido_al_girar", "fuga_aceite"]
TEXTO = "descripcion"
OBJETIVO = "falla"

# Palabras vacías del español (sin acentos, porque TF-IDF los elimina con strip_accents)
STOPWORDS_ES = """a al algo ante antes como con cuando de del desde donde el ella en entre era es esta
esto este ha hace hasta la las le lo los mas me mi muy no o para pero poco por que se si sin
sobre solo su sus tiene un una uno y ya favor buenas tardes ayuda""".split()

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


# --------------------------------------------------------------------------- datos
def cargar():
    df = pd.read_csv(RAIZ / "data" / "dataset_automuelles.csv")
    df[TEXTO] = df[TEXTO].fillna("")
    df["rango_antiguedad"] = pd.cut(2026 - df["anio"], [-1, 5, 12, 100],
                                    labels=["0-5 años", "6-12 años", "13+ años"]).astype(str)
    return df


def dividir(df):
    """70 % entrenamiento, 15 % validación, 15 % prueba, estratificado por clase."""
    tr, temp = train_test_split(df, test_size=0.30, stratify=df[OBJETIVO], random_state=SEMILLA)
    va, te = train_test_split(temp, test_size=0.50, stratify=temp[OBJETIVO], random_state=SEMILLA)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def preprocesador(con_texto=True):
    partes = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS),
        ("sin", "passthrough", SINTOMAS),
    ]
    if con_texto:
        partes.append(("txt", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2),
                                              stop_words=STOPWORDS_ES,
                                              min_df=3, sublinear_tf=True), TEXTO))
    return ColumnTransformer(partes)


def metricas(y, pred, proba, clases):
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1_macro": round(f1_score(y, pred, average="macro"), 4),
        "f1_ponderado": round(f1_score(y, pred, average="weighted"), 4),
        "top3_accuracy": round(top_k_accuracy_score(y, proba, k=3, labels=clases), 4),
    }


# --------------------------------------------------------------------------- modelos
def modelos_base():
    return {
        "Regresión logística (línea base)": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=SEMILLA),
        "LightGBM (sin ajustar)": LGBMClassifier(
            class_weight="balanced", random_state=SEMILLA, verbose=-1),
    }


def ajustar_lightgbm(tr, va, n_trials=60):
    """Búsqueda bayesiana (TPE) de hiperparámetros, maximizando F1-macro en validación."""
    prep = preprocesador().fit(tr)
    Xtr, Xva = prep.transform(tr), prep.transform(va)

    def objetivo(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        m = LGBMClassifier(**params, class_weight="balanced", random_state=SEMILLA, verbose=-1)
        m.fit(Xtr, tr[OBJETIVO])
        return f1_score(va[OBJETIVO], m.predict(Xva), average="macro")

    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    estudio.optimize(objetivo, n_trials=n_trials)
    return estudio


# --------------------------------------------------------------------------- equidad
def tabla_equidad(y, pred, grupos):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score,
                 "f1_macro": lambda a, b: f1_score(a, b, average="macro"),
                 "casos": lambda a, b: len(a)},
        y_true=y, y_pred=pred, sensitive_features=grupos)
    tabla = mf.by_group.copy()
    tabla["casos"] = tabla["casos"].astype(int)
    brecha = {"accuracy": round(float(mf.difference()["accuracy"]), 4),
              "f1_macro": round(float(mf.difference()["f1_macro"]), 4)}
    return tabla.round(4), brecha


def pesos_reponderados(df, columna):
    """Re-ponderación: cada grupo pesa lo mismo en total (peso inverso a su frecuencia)."""
    frec = df[columna].map(df[columna].value_counts(normalize=True))
    w = 1.0 / frec
    return (w / w.mean()).values


# --------------------------------------------------------------------------- gráficas
def grafica_comparacion(res):
    nombres = list(res)
    f1 = [res[n]["f1_macro"] for n in nombres]
    top3 = [res[n]["top3_accuracy"] for n in nombres]
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar(x - 0.2, f1, 0.4, label="F1-macro", color="#2b6cb0")
    b2 = ax.bar(x + 0.2, top3, 0.4, label="Top-3 accuracy", color="#9bc1e6")
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in nombres], fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Puntaje en validación")
    ax.set_title("Comparación de modelos (conjunto de validación)")
    ax.legend(loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "01_comparacion_modelos.png"); plt.close(fig)


def grafica_optuna(estudio):
    vals = [t.value for t in estudio.trials]
    mejor = np.maximum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(range(1, len(vals) + 1), vals, s=14, color="#9bc1e6", label="Prueba de Optuna")
    ax.plot(range(1, len(vals) + 1), mejor, color="#2b6cb0", lw=2, label="Mejor hasta el momento")
    ax.set_xlabel("Número de prueba"); ax.set_ylabel("F1-macro en validación")
    ax.set_title("Ajuste de hiperparámetros de LightGBM con Optuna")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "02_optuna_historial.png"); plt.close(fig)


def grafica_confusion(y, pred, clases):
    etiquetas = [c.replace("_", "\n") for c in clases]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=clases, display_labels=etiquetas,
                                            normalize="true", values_format=".2f",
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_xlabel("Falla predicha"); ax.set_ylabel("Falla real")
    ax.set_title("Matriz de confusión normalizada (conjunto de prueba)")
    plt.setp(ax.get_xticklabels(), fontsize=8); plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "03_matriz_confusion.png"); plt.close(fig)


def grafica_equidad(tablas, nombre_archivo, titulo):
    fig, axes = plt.subplots(1, len(tablas), figsize=(4.2 * len(tablas), 3.8), sharey=True)
    for ax, (atributo, tabla) in zip(axes, tablas.items()):
        barras = ax.bar(tabla.index.astype(str), tabla["f1_macro"], color="#2b6cb0")
        ax.bar_label(barras, labels=[f"{v:.2f}\n(n={n})" for v, n in
                                     zip(tabla["f1_macro"], tabla["casos"])], fontsize=7)
        ax.set_title(atributo.replace("_", " ").capitalize(), fontsize=10)
        ax.set_ylim(0, 1.15); ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1-macro en prueba")
    fig.suptitle(titulo); fig.tight_layout(); fig.savefig(FIG / nombre_archivo); plt.close(fig)


def grafica_importancia(pipeline, top=20):
    prep, modelo = pipeline.named_steps["prep"], pipeline.named_steps["modelo"]
    nombres = prep.get_feature_names_out()
    imp = pd.Series(modelo.booster_.feature_importance("gain"), index=nombres)
    imp = (imp / imp.sum()).sort_values().tail(top)
    etiquetas = [n.split("__", 1)[1] for n in imp.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(etiquetas, imp.values, color="#2b6cb0")
    ax.set_xlabel("Importancia relativa (ganancia)")
    ax.set_title(f"Las {top} variables más importantes del modelo final")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "05_importancia_variables.png"); plt.close(fig)


# --------------------------------------------------------------------------- principal
def main():
    df = cargar()
    tr, va, te = dividir(df)
    clases = sorted(df[OBJETIVO].unique())
    print(f"Entrenamiento={len(tr)}  Validación={len(va)}  Prueba={len(te)}")
    resultados = {"division": {"entrenamiento": len(tr), "validacion": len(va), "prueba": len(te)}}

    # 1. Comparación de modelos en validación ---------------------------------
    comparacion = {}
    for nombre, modelo in modelos_base().items():
        pipe = Pipeline([("prep", preprocesador()), ("modelo", modelo)]).fit(tr, tr[OBJETIVO])
        comparacion[nombre] = metricas(va[OBJETIVO], pipe.predict(va), pipe.predict_proba(va), clases)
        print(f"{nombre:35s} {comparacion[nombre]}")

    # Ablación: aporte de la descripción libre (TF-IDF)
    sin_txt = Pipeline([("prep", preprocesador(con_texto=False)),
                        ("modelo", LGBMClassifier(class_weight="balanced", random_state=SEMILLA,
                                                  verbose=-1))]).fit(tr, tr[OBJETIVO])
    resultados["ablacion_sin_texto"] = metricas(va[OBJETIVO], sin_txt.predict(va),
                                                sin_txt.predict_proba(va), clases)
    print("LightGBM sin texto (ablación)      ", resultados["ablacion_sin_texto"])

    # 2. Ajuste de hiperparámetros ---------------------------------------------
    estudio = ajustar_lightgbm(tr, va)
    mejores = estudio.best_params
    print("Mejores hiperparámetros:", mejores, "F1 val:", round(estudio.best_value, 4))
    final = Pipeline([("prep", preprocesador()),
                      ("modelo", LGBMClassifier(**mejores, subsample_freq=1, class_weight="balanced",
                                                random_state=SEMILLA, verbose=-1))])
    final.fit(tr, tr[OBJETIVO])
    comparacion["LightGBM (ajustado con Optuna)"] = metricas(
        va[OBJETIVO], final.predict(va), final.predict_proba(va), clases)
    resultados["comparacion_validacion"] = comparacion
    resultados["optuna"] = {"pruebas": len(estudio.trials),
                            "mejor_f1_validacion": round(estudio.best_value, 4),
                            "mejores_hiperparametros": mejores}

    # 3. Evaluación final en prueba (una sola vez) ----------------------------
    pred, proba = final.predict(te), final.predict_proba(te)
    resultados["prueba_modelo_final"] = metricas(te[OBJETIVO], pred, proba, clases)
    reporte = classification_report(te[OBJETIVO], pred, output_dict=True)
    resultados["reporte_por_clase"] = {c: {k: round(v, 4) for k, v in reporte[c].items()}
                                       for c in clases}
    print("PRUEBA:", resultados["prueba_modelo_final"])
    print(classification_report(te[OBJETIVO], pred, digits=3))

    # 4. Equidad -----------------------------------------------------------------
    atributos = ["tipo_vehiculo", "origen_marca", "rango_antiguedad"]
    equidad, tablas = {}, {}
    for a in atributos:
        tablas[a], brecha = tabla_equidad(te[OBJETIVO], pred, te[a])
        equidad[a] = {"por_grupo": tablas[a].to_dict(orient="index"), "brecha_max": brecha}
        print(f"\nEquidad por {a} (brecha F1={brecha['f1_macro']}):\n{tablas[a]}")
    resultados["equidad_modelo_final"] = equidad

    # Mitigación: re-ponderar para que cada tipo de vehículo pese lo mismo
    mitigado = Pipeline([("prep", preprocesador()),
                         ("modelo", LGBMClassifier(**mejores, subsample_freq=1,
                                                   class_weight="balanced",
                                                   random_state=SEMILLA, verbose=-1))])
    mitigado.fit(tr, tr[OBJETIVO], modelo__sample_weight=pesos_reponderados(tr, "tipo_vehiculo"))
    pred_m, proba_m = mitigado.predict(te), mitigado.predict_proba(te)
    tablas_m = {}
    equidad_m = {}
    for a in atributos:
        tablas_m[a], brecha = tabla_equidad(te[OBJETIVO], pred_m, te[a])
        equidad_m[a] = {"por_grupo": tablas_m[a].to_dict(orient="index"), "brecha_max": brecha}
    resultados["mitigacion_reponderacion"] = {
        "prueba": metricas(te[OBJETIVO], pred_m, proba_m, clases), "equidad": equidad_m}
    print("\nMITIGADO:", resultados["mitigacion_reponderacion"]["prueba"])
    print("Brecha tipo_vehiculo mitigada:", equidad_m["tipo_vehiculo"]["brecha_max"])

    # 5. Gráficas y artefactos ----------------------------------------------------
    grafica_comparacion(comparacion)
    grafica_optuna(estudio)
    grafica_confusion(te[OBJETIVO], pred, clases)
    grafica_equidad(tablas, "04_equidad_por_grupo.png", "Desempeño del modelo final por subgrupo")
    grafica_equidad({"modelo original": tablas["tipo_vehiculo"],
                     "modelo re-ponderado": tablas_m["tipo_vehiculo"]},
                    "06_mitigacion_tipo_vehiculo.png", "Efecto de la re-ponderación por tipo de vehículo")
    grafica_importancia(final)

    joblib.dump({"pipeline": final, "clases": clases}, RAIZ / "models" / "modelo_diagnostico.joblib")
    with open(RAIZ / "reports" / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    print("\nModelo guardado en models/modelo_diagnostico.joblib; resultados en reports/")


if __name__ == "__main__":
    main()
