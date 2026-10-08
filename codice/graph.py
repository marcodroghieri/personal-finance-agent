"""
graph.py — la pipeline con LangGraph, in sequenza:
estrazione (Node 1) -> validazione (Node 2) -> classificazione (Node 3)
-> assemblaggio del documento -> salvataggio nel registro.

Scelte semplici:
- Nessuna diramazione: se il Node 2 boccia il documento, la pipeline va avanti
  lo stesso. L'esito della validazione resta nello stato.
- Se il Node 1 fallisce, i passi dopo non fanno nulla e lo stato finale riporta
  errore_estrazione.
- Il salvataggio parte solo se salva_nel_registro è True. Un errore di scrittura
  non ferma la pipeline: finisce in errore_persistenza.

Da terminale:  python codice/graph.py <file_documento> [id]
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from node1_vision import estrai_documento
from node2_validator import valida
from node3_classifier import classifica
from persistenza import salva_documento
from pipeline_state import PipelineState


def nodo_estrazione(state: PipelineState) -> dict:
    """Node 1: legge il documento. Salva il risultato in documento_senza_categoria,
    oppure l'errore in errore_estrazione se tutti i tentativi falliscono."""
    documento, errore = estrai_documento(state["percorso_immagine"], state["id"])
    if errore is not None:
        return {"errore_estrazione": errore}
    return {"documento_senza_categoria": documento, "errore_estrazione": None}


def nodo_validazione(state: PipelineState) -> dict:
    """Node 2: controlla i numeri del documento (Python puro, senza modello)."""
    documento = state.get("documento_senza_categoria")
    if documento is None:
        # Estrazione fallita: niente da validare.
        return {}
    return {"risultato_validazione": valida(documento)}


def nodo_classificazione(state: PipelineState) -> dict:
    """Node 3: assegna la categoria. Classifica anche i documenti bocciati dal
    Node 2: che spesa sia e se i numeri tornino sono due domande separate."""
    documento = state.get("documento_senza_categoria")
    if documento is None:
        return {}
    risultato = classifica(
        esercente=documento.esercente,
        tipo_esercizio=documento.tipo_esercizio,
        voci=documento.voci,
        totale_stampato=documento.totale_stampato,
    )
    return {"risultato_classificazione": risultato}


def nodo_assemblaggio_finale(state: PipelineState) -> dict:
    """Costruisce il Documento completo, ora che la categoria è nota. Se manca
    il documento o la classificazione è fallita, documento_finale resta None."""
    documento = state.get("documento_senza_categoria")
    classificazione = state.get("risultato_classificazione")
    if documento is None or classificazione is None or classificazione.categoria is None:
        return {"documento_finale": None}
    return {"documento_finale": documento.con_categoria(classificazione.categoria)}


def nodo_persistenza(state: PipelineState) -> dict:
    """Salva il documento nel registro SQLite, ma solo se salva_nel_registro è
    True. Se non c'è un documento completo non fa nulla. Un errore di scrittura
    non ferma la pipeline: viene messo in errore_persistenza."""
    if not state.get("salva_nel_registro"):
        return {}

    documento_finale = state.get("documento_finale")
    if documento_finale is None:
        return {"salvato_nel_registro": False}

    classificazione = state.get("risultato_classificazione")
    motivazione = classificazione.motivazione if classificazione is not None else None

    try:
        salva_documento(
            documento_finale,
            risultato_validazione=state["risultato_validazione"],
            motivazione_categoria=motivazione,
            percorso_file=state.get("percorso_immagine"),
        )
        return {"salvato_nel_registro": True, "errore_persistenza": None}
    except Exception as e:  # noqa: BLE001 — qualunque errore di scrittura non deve fermare la pipeline
        return {
            "salvato_nel_registro": False,
            "errore_persistenza": f"{type(e).__name__}: {e}",
        }


def costruisci_grafo():
    """Costruisce il grafo: cinque passi uno dopo l'altro."""
    grafo = StateGraph(PipelineState)

    grafo.add_node("estrazione", nodo_estrazione)
    grafo.add_node("validazione", nodo_validazione)
    grafo.add_node("classificazione", nodo_classificazione)
    grafo.add_node("assemblaggio_finale", nodo_assemblaggio_finale)
    grafo.add_node("persistenza", nodo_persistenza)

    grafo.add_edge(START, "estrazione")
    grafo.add_edge("estrazione", "validazione")
    grafo.add_edge("validazione", "classificazione")
    grafo.add_edge("classificazione", "assemblaggio_finale")
    grafo.add_edge("assemblaggio_finale", "persistenza")
    grafo.add_edge("persistenza", END)

    return grafo.compile()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python graph.py <percorso_immagine_documento> [id_documento]")
        sys.exit(1)

    percorso_immagine = sys.argv[1]
    id_documento = sys.argv[2] if len(sys.argv) > 2 else "test_end_to_end_001"

    pipeline = costruisci_grafo()
    stato_finale = pipeline.invoke(
        {
            "id": id_documento,
            "percorso_immagine": percorso_immagine,
            # Lanciato a mano su un documento: lo si salva nel registro.
            "salva_nel_registro": True,
        }
    )

    print("=" * 64)
    print("STATO FINALE DELLA PIPELINE")
    print("=" * 64)

    if stato_finale.get("errore_estrazione"):
        print(f"\n[Node 1 - ESTRAZIONE FALLITA]\n  {stato_finale['errore_estrazione']}")
    else:
        print("\n[Node 1 - estrazione riuscita]")

    if "risultato_validazione" in stato_finale:
        print(f"\n[Node 2]\n{stato_finale['risultato_validazione']}")

    if "risultato_classificazione" in stato_finale:
        print(f"\n[Node 3]\n{stato_finale['risultato_classificazione']}")

    documento_finale = stato_finale.get("documento_finale")
    print("\n" + "=" * 64)
    if documento_finale is not None:
        print("RISULTATO COMPLETO (Documento):")
        print(documento_finale.model_dump_json(indent=2))
    else:
        print("NESSUN RISULTATO COMPLETO — la pipeline non è arrivata in fondo con un "
              "oggetto valido (vedi messaggi sopra per la causa: estrazione o classificazione falliti).")
    print("=" * 64)

    if stato_finale.get("errore_persistenza"):
        print(f"\n[Persistenza] salvataggio FALLITO: {stato_finale['errore_persistenza']}")
    elif stato_finale.get("salvato_nel_registro"):
        print("\n[Persistenza] documento salvato nel registro.")
    else:
        print("\n[Persistenza] niente da salvare (nessun risultato completo).")
