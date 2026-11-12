# Automuelles IA – Diagnóstico preliminar de fallas de suspensión

Proyecto de la **Actividad 4: Creación de un modelo de inteligencia artificial avanzada** (Electiva II – Inteligencia Artificial Avanzada).

El modelo orienta al cliente de *Automuelles Virtual*, una tienda de repuestos de suspensión. A partir de los datos del vehículo, los síntomas marcados y una descripción libre, propone las **tres fallas más probables** y los repuestos asociados. El diagnóstico es orientativo: siempre lo debe confirmar un técnico.

## Resultados (conjunto de prueba, 750 casos)

| Métrica | Valor |
| --- | --- |
| Exactitud | 67,7 % |
| F1-macro | 0,665 |
| Exactitud top-3 | 92,5 % |

- **Modelo final:** LightGBM ajustado con Optuna (60 pruebas). Se comparó contra una regresión logística (línea base) y un Random Forest.
- **Equidad:** se analizó con Fairlearn por tipo de vehículo, origen de la marca y antigüedad. El modelo acierta menos en vehículos de carga y de servicio público. La re-ponderación no cerró esa brecha, así que la API envía siempre esos casos a revisión técnica.

> **Limitación:** los datos son **semisintéticos** (5.000 casos generados con reglas técnicas de suspensión). Los resultados validan la metodología, no el desempeño en producción.

## Estructura

```
data/          dataset_automuelles.csv (5.000 casos)
src/
  generar_datos.py   genera el dataset semisintético
  entrenar.py        entrena, ajusta, evalúa, analiza equidad y guarda el modelo
  api.py             API FastAPI + formulario web de demostración
models/        modelo_diagnostico.joblib (pipeline entrenado)
reports/       resultados.json, log_entrenamiento.txt y figuras/
entregables/   informe (.docx) y presentación (.pptx)
```

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python src/generar_datos.py       # opcional: el dataset ya está incluido
python src/entrenar.py            # opcional: el modelo ya está incluido (tarda unos minutos)

uvicorn src.api:app --reload      # ejecutar desde la raíz del proyecto
```

Luego abra <http://127.0.0.1:8000> para usar el formulario, o <http://127.0.0.1:8000/docs> para ver la documentación de la API (Swagger).

Todo usa la semilla 42: al volver a generar los datos y entrenar se obtienen exactamente los mismos resultados.

### Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:8000/diagnostico -H "Content-Type: application/json" -d "{\"tipo_vehiculo\":\"carga\",\"marca\":\"Hino\",\"anio\":2010,\"kilometraje\":400000,\"carga_habitual\":\"alta\",\"sintomas\":[\"vehiculo_bajo\",\"inclinacion_lateral\"]}"
```

La respuesta trae las tres fallas más probables con su probabilidad y sus repuestos, el nivel de confianza y si el caso requiere revisión técnica.

## Tecnologías

Python · scikit-learn · LightGBM · Optuna · Fairlearn · FastAPI · pandas · matplotlib
