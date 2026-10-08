"""
Node 3 — Classificatore. Assegna a una spesa una categoria in base al tipo di
bene o servizio acquistato (Alimentari, Ristorazione, Trasporti, ecc.),
usando un modello di linguaggio locale (Ollama) in modalità solo testo: lavora
sui dati già estratti dal Node 1, non sull'immagine.

Perché un modello e non una tabella "tipo di negozio -> categoria"? Perché lo
stesso negozio vende cose diverse: in una farmacia i farmaci sono Salute, i
cosmetici sono Bellezza e cura personale. Bisogna guardare le voci.

classifica() riceve i campi del documento (non un oggetto Documento: la
categoria è proprio ciò che il Node 3 deve decidere). Se la risposta del modello
è tecnicamente sbagliata (JSON non valido, categoria che non esiste) riprova,
fino a max_tentativi volte. Non riprova per ottenere una categoria "diversa".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import ollama

from schema import CategoriaSpesa, Voce

# Qwen3.5 a volte lascia un blocco di ragionamento davanti alla risposta anche
# con think=False. Lo si toglie, altrimenti il JSON non si legge.
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_BLOCK_PATTERN = re.compile(
    r"Thinking\.\.\..*?\.\.\.\s*done thinking\.?", re.DOTALL | re.IGNORECASE
)


def _pulisci_residuo_thinking(testo: str) -> str:
    testo = _THINK_TAG_PATTERN.sub("", testo)
    testo = _THINKING_BLOCK_PATTERN.sub("", testo)
    return testo.strip()


MODELLO = "qwen3.5:9b"

# Descrizione di ogni categoria per il prompt: i criteri e i confini aiutano il
# modello nei casi ambigui (stesso negozio, beni diversi).
DESCRIZIONI_CATEGORIE = {
    CategoriaSpesa.ALIMENTARI: (
        "Generi alimentari e bevande da consumare a casa: supermercati, "
        "minimarket, alimentari, panetterie, macellerie, fruttivendoli. NON i "
        "pasti consumati fuori casa (quelli sono Ristorazione)."
    ),
    CategoriaSpesa.RISTORAZIONE: (
        "Pasti e consumazioni fuori casa: bar, caffetterie, ristoranti, "
        "pizzerie, trattorie, tavole calde, gastronomie con servizio, cibo da "
        "asporto e consegna a domicilio."
    ),
    CategoriaSpesa.TRASPORTI: (
        "Spostarsi e mantenere il veicolo: carburante e ricariche elettriche, "
        "biglietti di bus, metro, treno, aereo, traghetto, taxi e NCC, car e "
        "bike sharing, pedaggi, parcheggi; lavaggio auto, officina, gommista, "
        "tagliando, revisione, bollo e assicurazione del veicolo. Vale sia per "
        "gli spostamenti quotidiani sia per i viaggi: qualunque servizio che "
        "serva a muoversi va qui, a prescindere dallo scopo."
    ),
    CategoriaSpesa.CASA_UTENZE: (
        "La casa e le sue forniture ricorrenti: bollette di luce, gas, acqua, "
        "telefono fisso e mobile, internet; affitto e spese condominiali; "
        "arredamento, elettrodomestici grandi, articoli e accessori per la "
        "casa, ferramenta, materiali e interventi di manutenzione o "
        "ristrutturazione."
    ),
    CategoriaSpesa.SALUTE: (
        "Cura della salute: farmacia e parafarmacia, visite mediche e "
        "specialistiche, esami e analisi, prestazioni sanitarie, fisioterapia, "
        "dentista, occhiali e lenti, dispositivi medici."
    ),
    CategoriaSpesa.CURA_PERSONA: (
        "Estetica e igiene personale: parrucchiere, barbiere, estetista, "
        "centri benessere e SPA, prodotti per l'igiene, cosmetici, profumeria."
    ),
    CategoriaSpesa.ABBIGLIAMENTO_ACCESSORI: (
        "Capi e accessori da indossare: vestiti, scarpe, intimo, borse, "
        "cinture, cappelli, occhiali da sole, gioielli e bigiotteria, "
        "abbigliamento e calzature sportive."
    ),
    CategoriaSpesa.TEMPO_LIBERO: (
        "Svago, cultura, sport come attività, vacanze: palestra e abbonamenti "
        "sportivi, biglietti per eventi, concerti, cinema, teatro, musei, "
        "parchi e mostre, libri, riviste, giochi, hobby e relativa attrezzatura; "
        "alloggi e ingressi turistici in vacanza (hotel, b&b, attrazioni). Il "
        "trasporto per arrivarci resta però Trasporti."
    ),
    CategoriaSpesa.TECNOLOGIA_SERVIZI_DIGITALI: (
        "Dispositivi elettronici e servizi online: computer, smartphone, "
        "tablet, console, accessori e periferiche informatiche; software e app "
        "a pagamento; abbonamenti a streaming, cloud, servizi di AI; PEC, "
        "domini e hosting; ricariche di servizi digitali."
    ),
    CategoriaSpesa.ALTRO: (
        "Tutto ciò che non rientra chiaramente in nessuna categoria sopra: "
        "donazioni, multe, tasse e imposte, spese e commissioni bancarie, "
        "servizi generici, regali non identificabili come un tipo di bene "
        "preciso."
    ),
}


@dataclass
class RisultatoClassificazione:
    """Esito della classificazione di una spesa."""

    categoria: CategoriaSpesa | None  # None se la classificazione fallisce
    motivazione: str
    errore: str | None = None  # descrive il fallimento, se c'è stato

    def __str__(self) -> str:
        if self.errore:
            return f"[ERRORE] {self.errore}"
        return f"[{self.categoria.value}] {self.motivazione}"


def _costruisci_prompt(
    esercente: str, tipo_esercizio: str, voci: list[Voce], totale_stampato: float
) -> str:
    elenco_categorie = "\n".join(
        f"- {cat.value}: {descrizione}" for cat, descrizione in DESCRIZIONI_CATEGORIE.items()
    )
    voci_testo = "\n".join(f"  - {v.descrizione} ({v.prezzo:.2f}€)" for v in voci) or (
        "  (nessuna voce dettagliata: usa esercente, tipo di esercizio e totale)"
    )

    return f"""Sei un assistente di finanza personale. Classifichi una spesa in UNA categoria merceologica, cioè in base al TIPO DI BENE O SERVIZIO acquistato.

Categorie disponibili, coi relativi criteri:
{elenco_categorie}

Spesa da classificare:
- Esercente: {esercente}
- Tipo di esercizio: {tipo_esercizio}
- Voci:
{voci_testo}
- Totale: {totale_stampato:.2f}€

Regole:
- Scegli in base a COSA è stato acquistato, usando insieme il tipo di esercizio e le voci. Lo stesso esercente può ricadere in categorie diverse (una farmacia: farmaci → Salute, cosmetici → Bellezza e cura personale): guarda le voci.
- Se non ci sono voci dettagliate (bolletta, ricevuta di solo carburante), classifica in base a esercente e tipo di esercizio.
- Classifica per tipo di bene, non per destinatario: un regalo di abbigliamento è "Abbigliamento e accessori".
- Non farti guidare dalla somiglianza letterale tra le parole e il nome di una categoria: conta il significato reale della spesa.
- Usa "Altro" solo se davvero nessuna categoria si adatta.

Rispondi ESCLUSIVAMENTE con un oggetto JSON, senza testo aggiuntivo prima o dopo, senza markdown, nel formato:
{{"categoria": "<una delle etichette esatte sopra>", "motivazione": "<una frase breve>"}}
"""


def _pulisci_output_json(testo: str) -> str:
    """Toglie i blocchi markdown attorno al JSON, che il modello a volte aggiunge."""
    testo = testo.strip()
    if testo.startswith("```"):
        righe = testo.split("\n")
        righe = righe[1:]
        if righe and righe[-1].strip() == "```":
            righe = righe[:-1]
        testo = "\n".join(righe)
    return testo.strip()


def _ripara_graffa_finale(testo: str) -> str:
    """Se al JSON manca solo la graffa finale, la aggiunge. È successo davvero
    in due documenti: categoria e motivazione giuste, ma "}" omessa. Corregge
    solo la sintassi: se manca altro, il tentativo fallisce come prima."""
    testo = testo.rstrip()
    if testo.startswith("{") and not testo.endswith("}"):
        return testo + "}"
    return testo


def classifica(
    esercente: str,
    tipo_esercizio: str,
    voci: list[Voce],
    totale_stampato: float,
    max_tentativi: int = 2,
) -> RisultatoClassificazione:
    """Classifica una spesa e restituisce categoria e motivazione."""

    prompt = _costruisci_prompt(esercente, tipo_esercizio, voci, totale_stampato)
    ultimo_errore: RisultatoClassificazione | None = None

    for tentativo in range(1, max_tentativi + 1):
        try:
            risposta = ollama.chat(
                model=MODELLO,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                options={
                    "temperature": 0.1,
                    # 400 token bastano per il JSON. Il valore è esplicito perché
                    # il default di Ollama una volta ha troncato la risposta.
                    "num_predict": 400,
                },
            )
        except Exception as e:
            ultimo_errore = RisultatoClassificazione(
                categoria=None, motivazione="", errore=f"Errore di chiamata a Ollama: {e}"
            )
            continue

        testo_grezzo = _pulisci_residuo_thinking(risposta["message"]["content"])
        testo_pulito = _pulisci_output_json(testo_grezzo)

        try:
            try:
                dati = json.loads(testo_pulito)
            except json.JSONDecodeError:
                # Prima di scartare il tentativo si prova a riparare la graffa finale.
                riparato = _ripara_graffa_finale(testo_pulito)
                if riparato == testo_pulito:
                    raise
                dati = json.loads(riparato)
        except json.JSONDecodeError as e:
            ultimo_errore = RisultatoClassificazione(
                categoria=None,
                motivazione="",
                errore=(
                    f"[tentativo {tentativo}/{max_tentativi}] JSON non valido nella "
                    f"risposta del modello: {e}. Output grezzo: {testo_pulito!r}"
                ),
            )
            continue

        categoria_testo = dati.get("categoria")
        motivazione = dati.get("motivazione", "")

        try:
            categoria = CategoriaSpesa(categoria_testo)
        except ValueError:
            ultimo_errore = RisultatoClassificazione(
                categoria=None,
                motivazione=motivazione,
                errore=(
                    f"[tentativo {tentativo}/{max_tentativi}] Categoria '{categoria_testo}' "
                    f"restituita dal modello non è una delle categorie valide — "
                    f"possibile allucinazione o formulazione diversa dall'etichetta esatta"
                ),
            )
            continue

        return RisultatoClassificazione(categoria=categoria, motivazione=motivazione)

    # Tutti i tentativi falliti: si restituisce l'errore dell'ultimo.
    return ultimo_errore
