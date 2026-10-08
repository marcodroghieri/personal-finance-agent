"""
Stato condiviso della pipeline LangGraph (Node 1 -> 2 -> 3 -> assemblaggio -> salvataggio).
"""

from __future__ import annotations

from typing import Optional, TypedDict

from schema import Documento, DocumentoSenzaCategoria


class PipelineState(TypedDict, total=False):
    """Lo stato che passa da un nodo all'altro. Con total=False i campi si
    riempiono man mano: all'inizio ci sono solo id e percorso_immagine."""

    # Input
    id: str
    percorso_immagine: str

    # Salvataggio nel registro: parte solo se salva_nel_registro è True.
    salva_nel_registro: bool

    # Dopo il Node 1
    documento_senza_categoria: DocumentoSenzaCategoria
    errore_estrazione: Optional[str]

    # Dopo il Node 2 (RisultatoValidazione) e dopo il Node 3
    # (RisultatoClassificazione). Il tipo è object per evitare import circolari.
    risultato_validazione: object
    risultato_classificazione: object

    # Documento completo, o None se la pipeline non è arrivata in fondo
    documento_finale: Optional[Documento]

    # Esito del salvataggio. Se fallisce, la pipeline non si ferma: l'errore
    # resta in errore_persistenza.
    salvato_nel_registro: bool
    errore_persistenza: Optional[str]
