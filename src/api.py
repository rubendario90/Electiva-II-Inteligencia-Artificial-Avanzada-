"""
API de diagnóstico preliminar de Automuelles Virtual.

Uso:  .venv/Scripts/uvicorn src.api:app --reload
      Abrir http://127.0.0.1:8000  (formulario de demo)  o  /docs  (Swagger).

El resultado es una ORIENTACIÓN: siempre debe confirmarlo un técnico.
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
_art = joblib.load(RAIZ / "models" / "modelo_diagnostico.joblib")
MODELO, CLASES = _art["pipeline"], _art["clases"]

SINTOMAS = ["rebote_excesivo", "golpe_en_huecos", "inclinacion_lateral", "vehiculo_bajo",
            "desgaste_irregular_llantas", "vibracion_volante", "juego_en_direccion",
            "ruido_al_girar", "fuga_aceite"]

ORIGEN = {"BYD": "china", "Chery": "china", "JAC": "china", "Great Wall": "china", "Foton": "china"}

NOMBRES = {
    "amortiguador_danado": "Amortiguador dañado",
    "muelle_fatigado": "Muelle (ballesta) fatigado o partido",
    "espiral_vencido": "Espiral vencido",
    "bujes_desgastados": "Bujes desgastados",
    "terminal_rotula": "Terminal de dirección o rótula",
    "sin_falla_suspension": "Sin falla aparente de suspensión",
}

# Catálogo de repuestos sugeridos por falla (se reemplaza por el catálogo real de la tienda)
REPUESTOS = {
    "amortiguador_danado": ["Amortiguador delantero/trasero (par)", "Kit de guardapolvo y tope", "Base de amortiguador"],
    "muelle_fatigado": ["Hoja maestra de muelle", "Paquete de muelle completo", "Abrazaderas y grilletes"],
    "espiral_vencido": ["Espiral reforzado (par)", "Aislador de espiral"],
    "bujes_desgastados": ["Kit de bujes de tijera", "Bujes de barra estabilizadora", "Bujes de muelle"],
    "terminal_rotula": ["Terminal de dirección", "Rótula inferior", "Axial de dirección"],
    "sin_falla_suspension": ["Revisión de alineación y balanceo", "Revisión de presión de llantas"],
}


class Solicitud(BaseModel):
    tipo_vehiculo: Literal["particular", "camioneta", "servicio_publico", "carga"]
    marca: str = Field(examples=["Chevrolet"])
    anio: int = Field(ge=1980, le=2027, examples=[2015])
    kilometraje: int = Field(ge=0, examples=[120000])
    carga_habitual: Literal["baja", "media", "alta"] = "media"
    tipo_via: Literal["urbana", "mixta", "destapada"] = "urbana"
    sintomas: list[str] = Field(default_factory=list, examples=[["rebote_excesivo", "fuga_aceite"]])
    descripcion: str = ""


app = FastAPI(title="Automuelles Virtual – Diagnóstico preliminar de suspensión")


@app.post("/diagnostico")
def diagnosticar(s: Solicitud):
    fila = {
        "tipo_vehiculo": s.tipo_vehiculo, "marca": s.marca,
        "origen_marca": ORIGEN.get(s.marca, "tradicional"),
        "anio": s.anio, "kilometraje": s.kilometraje,
        "carga_habitual": s.carga_habitual, "tipo_via": s.tipo_via,
        **{k: int(k in s.sintomas) for k in SINTOMAS},
        "descripcion": s.descripcion,
    }
    proba = MODELO.predict_proba(pd.DataFrame([fila]))[0]
    orden = proba.argsort()[::-1][:3]
    top3 = [{"falla": CLASES[i], "nombre": NOMBRES[CLASES[i]],
             "probabilidad": round(float(proba[i]), 3), "repuestos": REPUESTOS[CLASES[i]]}
            for i in orden]
    confianza = top3[0]["probabilidad"]
    return {
        "diagnosticos_probables": top3,
        "confianza": "alta" if confianza >= 0.6 else "media" if confianza >= 0.4 else "baja",
        "advertencia": ("Diagnóstico orientativo generado por IA. Antes de comprar o instalar, "
                        "confírmelo con un técnico de Automuelles."),
        "requiere_revision_tecnica": confianza < 0.4 or s.tipo_vehiculo == "carga",
    }


FORM = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Automuelles Virtual · Diagnóstico IA</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#1a202c;background:#f7fafc}
h1{color:#2b6cb0;font-size:1.5rem}fieldset{border:1px solid #cbd5e0;border-radius:8px;margin:12px 0;background:#fff}
label{display:block;margin:6px 0}select,input,textarea{padding:6px;width:100%;box-sizing:border-box}
.chk label{display:inline-block;width:48%}.chk input{width:auto}
button{background:#2b6cb0;color:#fff;border:0;padding:10px 18px;border-radius:6px;font-size:1rem;cursor:pointer}
.card{background:#fff;border-left:5px solid #2b6cb0;padding:10px 14px;margin:8px 0;border-radius:6px}
.bar{height:8px;background:#2b6cb0;border-radius:4px}.warn{background:#fffbea;border:1px solid #f6e05e;padding:10px;border-radius:6px}
</style></head><body>
<h1>Automuelles Virtual · Diagnóstico preliminar de suspensión</h1>
<form id="f"><fieldset><legend>Vehículo</legend>
<label>Tipo <select name="tipo_vehiculo"><option value="particular">Particular</option>
<option value="camioneta">Camioneta</option><option value="servicio_publico">Servicio público</option>
<option value="carga">Carga</option></select></label>
<label>Marca <select name="marca">MARCAS</select></label>
<label>Año <input name="anio" type="number" value="2015"></label>
<label>Kilometraje <input name="kilometraje" type="number" value="120000"></label>
<label>Carga habitual <select name="carga_habitual"><option>baja</option><option selected>media</option><option>alta</option></select></label>
<label>Tipo de vía <select name="tipo_via"><option>urbana</option><option>mixta</option><option>destapada</option></select></label>
</fieldset><fieldset class="chk"><legend>Síntomas</legend>SINTOMAS</fieldset>
<fieldset><legend>Cuéntenos con sus palabras</legend><textarea name="descripcion" rows="3"
placeholder="Ej.: el carro rebota mucho después de un resalto"></textarea></fieldset>
<button>Diagnosticar</button></form><div id="r"></div>
<script>
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const d=new FormData(e.target),b=Object.fromEntries(d);b.sintomas=d.getAll('sintomas');
b.anio=+b.anio;b.kilometraje=+b.kilometraje;
const r=await (await fetch('/diagnostico',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json();
let h='<h2>Resultado (confianza '+r.confianza+')</h2>';
r.diagnosticos_probables.forEach(x=>{h+='<div class="card"><b>'+x.nombre+'</b> – '+(x.probabilidad*100).toFixed(1)+
'%<div class="bar" style="width:'+(x.probabilidad*100)+'%"></div><small>Repuestos sugeridos: '+x.repuestos.join(', ')+'</small></div>'});
h+='<p class="warn">⚠ '+r.advertencia+(r.requiere_revision_tecnica?' <b>Se recomienda revisión presencial.</b>':'')+'</p>';
document.getElementById('r').innerHTML=h;};
</script></body></html>"""

MARCAS = ["Chevrolet", "Renault", "Mazda", "Kia", "Toyota", "Nissan", "Hyundai", "Suzuki", "Ford",
          "Mitsubishi", "Hino", "Isuzu", "International", "BYD", "Chery", "JAC", "Great Wall", "Foton"]


@app.get("/", response_class=HTMLResponse)
def formulario():
    marcas = "".join(f"<option>{m}</option>" for m in MARCAS)
    sint = "".join(f'<label><input type="checkbox" name="sintomas" value="{s}"> '
                   f'{s.replace("_", " ").capitalize()}</label>' for s in SINTOMAS)
    return FORM.replace("MARCAS", marcas).replace("SINTOMAS", sint)
