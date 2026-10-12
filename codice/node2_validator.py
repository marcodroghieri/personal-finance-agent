"""
Node 2 — Validatore. Controlla che i numeri di un documento siano coerenti.
Python puro, nessun modello: sullo stesso documento dà sempre lo stesso esito.

Controllo (sui documenti con voci): la somma dei prezzi delle voci è uguale
al totale stampato. Il documento riporta due volte la stessa quantità, nelle
voci e nel totale: se non coincidono, una delle due letture è sbagliata.

Vale anche per le bollette, le cui voci sono le righe del riepilogo degli
importi. Un documento senza voci (ricevuta POS, bolletta senza riepilogo) non
si può sommare: il risultato riporta un avviso di "validazione parziale".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schema import DocumentoSenzaCategoria, TipoDocumento

# Tolleranza in euro. Non si confronta con l'uguaglianza esatta perché il
# registratore di cassa arrotonda al centesimo riga per riga.
TOLLERANZA_TOTALE_EUR = 0.02


@dataclass
class RisultatoValidazione:
    """Esito della validazione di un documento."""

    valido: bool
    errori: list[str] = field(default_factory=list)  # problemi che rendono il documento non valido
    avvisi: list[str] = field(default_factory=list)  # segnalazioni che non lo bocciano

    def __str__(self) -> str:
        stato = "VALIDO" if self.valido else "NON VALIDO"
        righe = [f"[{stato}]"]
        for e in self.errori:
            righe.append(f"  ERRORE: {e}")
        for a in self.avvisi:
            righe.append(f"  avviso: {a}")
        return "\n".join(righe)


def _scarto_al_centesimo(a: float, b: float) -> float:
    """Differenza fra due importi, arrotondata al centesimo. Serve perché i
    decimali del computer non sono esatti: 10,05 - 10,03 dà
    0,02000000000000135, che supererebbe la tolleranza di 0,02."""
    return round(abs(a - b), 2)


def valida_somma_voci(documento: DocumentoSenzaCategoria) -> list[str]:
    """La somma dei prezzi delle voci deve essere uguale al totale stampato.
    Senza voci non c'è nulla da sommare: nessun errore (l'avviso lo aggiunge valida())."""
    errori = []
    if not documento.voci:
        return errori
    somma_voci = sum(v.prezzo for v in documento.voci)
    scarto = _scarto_al_centesimo(somma_voci, documento.totale_stampato)
    if scarto > TOLLERANZA_TOTALE_EUR:
        errori.append(
            f"La somma delle voci ({somma_voci:.2f}€) non corrisponde al totale stampato "
            f"({documento.totale_stampato:.2f}€) — scarto di {scarto:.2f}€"
        )
    return errori


def valida(documento: DocumentoSenzaCategoria) -> RisultatoValidazione:
    """Esegue il controllo e restituisce l'esito."""
    errori = valida_somma_voci(documento)

    avvisi = []
    if not documento.voci:
        if documento.tipo_documento == TipoDocumento.BOLLETTA:
            avvisi.append(
                "Documento senza voci (bolletta): validazione parziale — nessun controllo "
                "numerico applicabile, la somma delle voci richiede le voci."
            )
        else:
            avvisi.append(
                f"Documento di tipo '{documento.tipo_documento.value}' senza nessuna voce estratta: "
                "possibile estrazione incompleta del Node 1 — nessun controllo numerico applicabile."
            )

    return RisultatoValidazione(valido=(len(errori) == 0), errori=errori, avvisi=avvisi)
