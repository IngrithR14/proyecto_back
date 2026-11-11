import pandas as pd
from app.core.seudonimizar import seudonimizar, ELIMINAR

def test_seudonimizar_sintetico(tmp_path, monkeypatch):
    import app.core.seudonimizar as s
    monkeypatch.setattr(s, "PRIVADO", tmp_path / "priv.json")
    df = pd.DataFrame({"CodigoCliente": [1, 2, 3], "Supervisor": ["ana", "ana", None],
                       "Identificacion": [9, 9, 8], "Direccion": ["x"] * 3,
                       "Cliente": ["a", "b", "c"]})
    # usa nombres que existan en asesores.json; con nombres desconocidos el mapa los acepta igual
    out = seudonimizar(df)
    assert not (set(out.columns) & set(ELIMINAR))
    assert set(out["equipo_codigo"]) <= {"EQ00", "EQ01"}