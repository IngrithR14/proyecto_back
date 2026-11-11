from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
from .asesores import _norm, normalizar_equipo

PRIVADO = Path(os.getenv("ASESORES_PRIVADO", str(
    Path(__file__).resolve().parent.parent / "knowledge" / "asesores_privado.json")))
COL_ORIGEN = "Supervisor"
SIN = "SIN_ASESOR"
ELIMINAR = ["Identificacion", "Cliente", "Direccion", "Propietario", "Administrador",
            "Telefonos", "CorreoElectronico", "TelSupervisor", COL_ORIGEN]
N_EQUIPOS_ESPERADO = 6


def _mapa() -> dict[str, str]:
    return json.loads(PRIVADO.read_text(encoding="utf-8")) if PRIVADO.exists() else {}


def seudonimizar(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza Supervisor por equipo_codigo (EQ01..EQ06, EQ00=SIN_ASESOR) y
    elimina columnas personales. El mapa nombre->EQ se congela en el archivo privado."""
    out = normalizar_equipo(df, col=COL_ORIGEN)              # repara U+FFFD primero
    claves = out[COL_ORIGEN].fillna(SIN).map(_norm)
    sin = _norm(SIN)
    mapa = _mapa()
    for n in sorted(set(claves) - {sin} - set(mapa)):        # nuevos al final, sin renumerar
        mapa[n] = f"EQ{len(mapa) + 1:02d}"
    if len(mapa) > N_EQUIPOS_ESPERADO:
        raise ValueError(f"{len(mapa)} equipos (esperado {N_EQUIPOS_ESPERADO}): "
                         "hay variantes de nombre sin normalizar")
    PRIVADO.write_text(json.dumps(mapa, ensure_ascii=False, indent=1), encoding="utf-8")
    out["equipo_codigo"] = claves.map(lambda k: "EQ00" if k == sin else mapa[k])
    return out.drop(columns=[c for c in ELIMINAR if c in out.columns])