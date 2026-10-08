"""
chat.py — L'interfaccia di chat (Chainlit).

Una sola finestra per tutto:
- un messaggio con un file (foto o PDF) va alla pipeline (graph.py): il documento
  viene letto, controllato, classificato e, se tutto torna, salvato nel registro;
- un messaggio di solo testo è una domanda e va all'agente (agente_interrogazione.py);
- la parola "riepilogo" mostra il grafico della spesa per categoria.

La scelta tra pipeline e agente la fa il programma guardando se c'è un file
allegato, non il modello: non può sbagliare.

Avvio:  chainlit run codice/chat.py     (oppure doppio clic su avvia_chat.bat)
Serve Ollama con il modello qwen3.5:9b.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import chainlit as cl

import matplotlib
matplotlib.use("Agg")  # disegna su immagine, senza aprire finestre
import matplotlib.pyplot as plt

from agente_interrogazione import esegui_query_sicura, interroga
from graph import costruisci_grafo
from persistenza import salva_documento

# Se True, i documenti caricati in chat vengono salvati nel registro
# (registro_spese.db). Con False la chat elabora e mostra il risultato ma non
# scrive nulla: utile per provare senza sporcare il registro vero.
SALVA_NEL_REGISTRO = False

PIPELINE = costruisci_grafo()

# Formati di file accettati.
ESTENSIONI_VALIDE = {".jpg", ".jpeg", ".png", ".pdf"}


@cl.on_chat_start
async def avvio() -> None:
    await cl.Message(
        content=(
            "Ciao! Carica una foto o un PDF di uno scontrino, ricevuta, fattura, "
            "bolletta o biglietto: lo leggo, controllo i numeri, lo classifico e "
            "lo salvo nel registro.\n\n"
            "Oppure fammi una domanda sulle tue spese, per esempio "
            "\"quanto ho speso per ogni categoria?\".\n\n"
            "Scrivi \"riepilogo\" per vedere il grafico della spesa per categoria."
        )
    ).send()


@cl.on_message
async def ricevi_messaggio(messaggio: cl.Message) -> None:
    """Gestisce un messaggio: con un file va alla pipeline, con solo testo
    all'agente. Di un allegato servono solo il percorso locale (.path) e
    l'estensione, qualunque sia il tipo di elemento di Chainlit."""
    allegati = [
        elemento for elemento in messaggio.elements
        if getattr(elemento, "path", None)
        and Path(elemento.path).suffix.lower() in ESTENSIONI_VALIDE
    ]

    # Instradamento: file -> pipeline, solo testo -> agente di interrogazione.
    if allegati:
        for allegato in allegati:
            await _elabora_documento(allegato)
        # Il testo scritto insieme a un file spesso è un commento ("ecco lo
        # scontrino"), non una domanda: non va all'agente, e lo si dice all'utente.
        if (messaggio.content or "").strip():
            await cl.Message(
                content=(
                    "Ho elaborato il documento. Il testo scritto insieme al file non "
                    "l'ho usato come domanda: se vuoi chiedermi qualcosa, scrivilo "
                    "in un messaggio a parte."
                )
            ).send()
        return

    if messaggio.elements:
        # Allegato di un formato che non si legge.
        await cl.Message(
            content="Questo formato non lo leggo. Carica una foto (jpg/png) o un PDF."
        ).send()
        return

    domanda = (messaggio.content or "").strip()
    if not domanda:
        return
    # "riepilogo" mostra il grafico (anche questa scelta la fa il programma).
    if domanda.lower() == "riepilogo":
        await _mostra_riepilogo()
        return
    await _rispondi_domanda(domanda)


# Query fissa, scritta a mano (non generata dal modello). Usa la stessa vista
# dell'agente, quindi i numeri coincidono con le sue risposte.
QUERY_RIEPILOGO = """
SELECT categoria, SUM(spesa) AS totale
FROM documenti_spesa
GROUP BY categoria
ORDER BY totale DESC
"""

async def _mostra_riepilogo() -> None:
    """Mostra un grafico a barre della spesa per categoria. La query passa
    per esegui_query_sicura dell'agente, cioè in sola lettura."""
    righe, errore = esegui_query_sicura(QUERY_RIEPILOGO)
    if errore:
        await cl.Message(content=f"Non sono riuscito a leggere il registro: {errore}").send()
        return
    if not righe:
        await cl.Message(content="Il registro è vuoto: non c'è niente da riassumere.").send()
        return

    # Barre orizzontali perché i nomi delle categorie sono lunghi. Le liste si
    # invertono perché matplotlib disegna la prima barra in basso.
    categorie = [r["categoria"] for r in righe][::-1]
    totali = [r["totale"] or 0 for r in righe][::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(categorie, totali)
    ax.set_xlabel("Spesa (€)")
    ax.set_title("Spesa per categoria")
    for y, valore in enumerate(totali):
        ax.text(valore, y, f" {_formatta_valore('totale', valore)}", va="center")
    ax.set_xlim(0, max(totali) * 1.25)  # spazio per le etichette
    fig.tight_layout()

    await cl.Message(
        content=f"Spesa totale nel registro: {_formatta_valore('totale', sum(totali))}",
        elements=[cl.Pyplot(name="riepilogo", figure=fig, display="inline", size="large")],
    ).send()
    plt.close(fig)


async def _rispondi_domanda(domanda: str) -> None:
    """Passa la domanda all'agente. interroga() è sincrona e aspetta Ollama:
    make_async la esegue in un thread, così la chat non si blocca."""
    await cl.Message(content="Cerco nel registro...").send()
    risultato = await cl.make_async(interroga)(domanda)

    if risultato.errore:
        # Il dettaglio tecnico va nel terminale, all'utente un messaggio semplice.
        print(f"[chat] domanda senza risposta: {domanda!r}\n       {risultato.errore}")
        await cl.Message(
            content="Non sono riuscito a rispondere a questa domanda. "
            "Prova a riformularla, per esempio indicando il prodotto, "
            "l'esercente o il mese."
        ).send()
        return

    # La risposta si formatta qui, in testo semplice, dalle righe grezze.
    await cl.Message(content=_formatta_righe(risultato.righe or [])).send()


# Parole che, nel nome di una colonna, fanno pensare a un importo in euro. È
# un'euristica: il nome lo sceglie il modello. Quelle che indicano un conteggio
# hanno la precedenza.
_PAROLE_IMPORTO = ("spes", "totale", "prezzo", "importo", "costo", "euro", "somma", "media")
_PAROLE_CONTEGGIO = ("numero", "conteggio", "count", "quanti", "quante", "num_", "n_")

MASSIMO_RIGHE_MOSTRATE = 20


def _e_importo(colonna: str) -> bool:
    nome = colonna.lower()
    if any(p in nome for p in _PAROLE_CONTEGGIO):
        return False
    return any(p in nome for p in _PAROLE_IMPORTO)


def _nome_leggibile(colonna: str) -> str:
    """totale_spesa -> Totale spesa."""
    testo = colonna.replace("_", " ").strip()
    return testo[:1].upper() + testo[1:]


def _formatta_valore(colonna: str, valore) -> str:
    """Numeri in formato italiano (13.035,69), con € solo se la colonna sembra
    un importo. Testi e date restano come sono."""
    if valore is None:
        return "—"
    if isinstance(valore, bool) or not isinstance(valore, (int, float)):
        return str(valore)
    if _e_importo(colonna):
        numero = f"{valore:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{numero} €"
    if isinstance(valore, float) and not valore.is_integer():
        return f"{valore:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{int(valore):,}".replace(",", ".")


def _formatta_righe(righe: list[dict]) -> str:
    """Un solo valore -> una frase; più righe -> un elenco, una riga ciascuna
    (es. "Casa e utenze: 8.057,79 €")."""
    if not righe:
        return "Nessun risultato trovato nel registro per questa domanda."

    colonne = list(righe[0].keys())
    if len(righe) == 1 and len(colonne) == 1:
        colonna = colonne[0]
        # Un SUM su zero righe vale NULL, non 0: lo si dice a parole.
        if righe[0][colonna] is None:
            return "Nessuna spesa trovata nel registro per questa richiesta."
        return f"**{_nome_leggibile(colonna)}:** {_formatta_valore(colonna, righe[0][colonna])}"

    # La prima colonna fa da etichetta (es. la categoria), le altre seguono.
    etichetta, altre = colonne[0], colonne[1:]
    elenco = []
    for riga in righe[:MASSIMO_RIGHE_MOSTRATE]:
        if not altre:
            elenco.append(f"- {_formatta_valore(etichetta, riga[etichetta])}")
            continue
        valori = ", ".join(
            _formatta_valore(c, riga[c]) if len(altre) == 1
            else f"{_nome_leggibile(c).lower()} {_formatta_valore(c, riga[c])}"
            for c in altre
        )
        elenco.append(f"- {_formatta_valore(etichetta, riga[etichetta])}: {valori}")
    testo = "\n".join(elenco)
    if len(righe) > MASSIMO_RIGHE_MOSTRATE:
        testo += f"\n\n_…e altre {len(righe) - MASSIMO_RIGHE_MOSTRATE} righe, non mostrate._"
    return testo


async def _elabora_documento(allegato) -> None:
    percorso = allegato.path
    # Id univoco per ogni file caricato (non il nome del file), per non confondersi
    # con gli id doc_01..doc_75 del dataset.
    id_documento = f"chat_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    await cl.Message(content=f"Elaboro **{allegato.name}**...").send()

    try:
        # La pipeline aspetta Ollama, quindi gira in un thread (come interroga()).
        # Qui non salva mai: se salvare lo decide _decidi_salvataggio, dopo,
        # perché per i documenti bocciati dal Node 2 serve prima la risposta dell'utente.
        stato_finale = await cl.make_async(PIPELINE.invoke)({
            "id": id_documento,
            "percorso_immagine": percorso,
            "salva_nel_registro": False,
        })
    except Exception as e:  # noqa: BLE001 — la chat non deve crollare per un errore imprevisto
        await cl.Message(content=f"Errore imprevisto nell'elaborazione: {type(e).__name__}: {e}").send()
        return

    await cl.Message(content=_formatta_risultato(stato_finale)).send()
    await _decidi_salvataggio(stato_finale)


async def _decidi_salvataggio(stato: dict) -> None:
    """Documento valido -> si salva senza domande. Documento bocciato dal Node 2
    -> la chat chiede all'utente con due pulsanti. Se non risponde entro il tempo
    limite il documento NON si salva: nel dubbio meglio non scrivere nel registro."""
    documento = stato.get("documento_finale")
    if documento is None:
        return  # niente da salvare (l'errore è già stato mostrato)

    validazione = stato.get("risultato_validazione")
    if validazione is not None and not validazione.valido:
        risposta = await cl.AskActionMessage(
            content="I conti di questo documento non tornano. Lo salvo lo stesso nel registro?",
            actions=[
                cl.Action(name="salva", payload={"scelta": "salva"}, label="Salva comunque"),
                cl.Action(name="scarta", payload={"scelta": "scarta"}, label="Scarta"),
            ],
            timeout=120,
        ).send()
        # Si legge "name" dell'azione scelta (il formato del payload cambia tra le
        # versioni di Chainlit). risposta è None se il tempo scade.
        scelta = (risposta or {}).get("name")
        if scelta != "salva":
            testo = "Documento scartato." if scelta == "scarta" else "Nessuna risposta: documento non salvato."
            await cl.Message(content=f"{testo} Non è finito nel registro.").send()
            return

    if not SALVA_NEL_REGISTRO:
        await cl.Message(
            content="_(salvataggio disattivato per questo test — non è finito nel registro)_"
        ).send()
        return

    classificazione = stato.get("risultato_classificazione")
    try:
        # Stessa chiamata del nodo di salvataggio di graph.py. Se il documento era
        # stato bocciato dal Node 2, node2_valido vale 0.
        await cl.make_async(salva_documento)(
            documento,
            risultato_validazione=validazione,
            motivazione_categoria=getattr(classificazione, "motivazione", None),
            percorso_file=stato.get("percorso_immagine"),
        )
    except Exception as e:  # noqa: BLE001 — la chat non deve crollare per un errore di scrittura
        await cl.Message(content=f"Salvataggio nel registro FALLITO: {type(e).__name__}: {e}").send()
        return

    await cl.Message(content="Salvato nel registro.").send()


def _formatta_risultato(stato: dict) -> str:
    """Trasforma lo stato finale della pipeline in un messaggio leggibile."""
    if stato.get("errore_estrazione"):
        return f"Non sono riuscito a leggere il documento: {stato['errore_estrazione']}"

    documento = stato.get("documento_finale")
    if documento is None:
        return (
            "Il documento è stato letto ma non sono arrivato a un risultato completo "
            "(classificazione non riuscita)."
        )

    validazione = stato.get("risultato_validazione")
    if validazione is not None and validazione.valido:
        riga_validazione = "controllo numeri: OK" + (
            f" (avvisi: {'; '.join(validazione.avvisi)})" if validazione.avvisi else ""
        )
    elif validazione is not None:
        riga_validazione = f"controllo numeri: DA VERIFICARE ({'; '.join(validazione.errori)})"
    else:
        riga_validazione = "controllo numeri: non eseguito"

    righe = [
        f"**Esercente:** {documento.esercente}",
        f"**Categoria:** {documento.categoria.value}",
    ]

    # La motivazione del Node 3 si mostra subito, così l'utente si accorge di una
    # categoria sbagliata nel momento del caricamento.
    classificazione = stato.get("risultato_classificazione")
    if classificazione is not None and getattr(classificazione, "motivazione", ""):
        righe.append(f"_Perché: {classificazione.motivazione}_")

    righe += [
        f"**Totale:** {documento.totale_stampato:.2f}€",
        riga_validazione,
        "",
        _tabella_voci(documento.voci),
    ]
    # L'esito del salvataggio non è qui: lo comunica _decidi_salvataggio.
    return "\n".join(righe)


def _tabella_voci(voci) -> str:
    """Le voci del documento come tabella Markdown. Se non ci sono voci
    (bollette, ricevute POS) lo dice, invece di mostrare una tabella vuota."""
    if not voci:
        return "_Il documento non riporta voci singole._"

    tabella = [
        "| Voce | Prezzo | IVA |",
        "|---|---:|---:|",
    ]
    for voce in voci:
        # Il carattere | romperebbe la tabella Markdown: lo si sostituisce.
        descrizione = voce.descrizione.replace("|", "/")
        aliquota = "—" if voce.iva_aliquota is None else f"{voce.iva_aliquota:g}%"
        tabella.append(f"| {descrizione} | {voce.prezzo:.2f}€ | {aliquota} |")
    return "\n".join(tabella)
