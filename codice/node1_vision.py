"""
Node 1 — Lettura del documento. Prende un file (foto o PDF) e restituisce un
DocumentoSenzaCategoria con esercente, data, voci, totale e IVA, usando un
modello locale (Ollama).

Come viene dato il documento al modello (_prepara_documento):
- foto (JPEG, PNG, ...)         -> modello con vista, una immagine;
- PDF con testo                 -> si estrae il testo di tutte le pagine e lo si
                                   dà al modello come testo (nelle bollette la
                                   prima pagina può essere pubblicità);
- PDF scansionato (senza testo) -> modello con vista, fino a 6 pagine come immagini.

Se la risposta non si può usare (chiamata fallita, JSON non valido, campi
mancanti) il modello viene interrogato di nuovo, fino a max_tentativi volte.
Se invece l'estrazione è valida ma il Node 2 la boccia, il modello riceve in
conversazione l'errore preciso e può correggersi. Alla fine si tiene il
tentativo con lo scarto minore tra somma delle voci e totale, non per forza
l'ultimo.

Le bollette si leggono in due chiamate: la prima, uguale per tutti i documenti,
legge esercente, data e totale; la seconda (_voci_bolletta) legge le voci del
riepilogo degli importi, e le tiene solo se la loro somma torna con il totale.
"""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF: legge il testo dei PDF e li converte in immagini
import ollama
from PIL import Image

from node2_validator import valida
from schema import DocumentoSenzaCategoria, TipoDocumento, Voce

MODELLO = "qwen3.5:9b"
LATO_MASSIMO_PX = 1568
TEMPERATURE = 0.1
MAX_TENTATIVI_DEFAULT = 2

# Risoluzione (DPI) con cui un PDF viene convertito in immagine.
PDF_DPI = 200

# Un PDF con almeno questi caratteri di testo è "digitale" e si passa al
# modello come testo; sotto questa soglia è una scansione e si usano le immagini.
SOGLIA_TESTO_PDF = 100

# Pagine al massimo da convertire in immagine per un PDF scansionato.
MAX_PAGINE_PDF_SCAN = 6

# Prompt di estrazione. Le regole sulle voci dello scontrino vengono prima, i
# casi particolari (bollette, ecc.) sono in un blocco a parte: una versione più
# lunga, con tutte le regole mescolate, faceva peggiorare la lettura degli scontrini.
PROMPT_ESTRAZIONE = """Estrai i dati di un documento di spesa italiano (scontrino, ricevuta, fattura, bolletta, biglietto) e restituisci ESCLUSIVAMENTE questo oggetto JSON, senza testo prima o dopo, senza markdown, senza commenti:

{
  "tipo_documento": "scontrino | ricevuta | fattura | bolletta | biglietto | altro",
  "esercente": "chi emette il documento (negozio, fornitore, vettore)",
  "tipo_esercizio": "es. supermercato, ristorante, abbigliamento, utenza gas, compagnia aerea",
  "data": "YYYY-MM-DD, oppure null se non leggibile con certezza",
  "voci": [
    {"descrizione": "...", "prezzo": 0.00, "iva_aliquota": 22}
  ],
  "totale_stampato": 0.00,
  "iva_totale_stampata": 0.00
}

REGOLE SULLE VOCI (righe di spesa):
- Ogni riga di prodotto ha UN prezzo, quello stampato accanto alla descrizione. Riporta quel valore esattamente, senza fare calcoli o correzioni.
- NON unire mai due righe di prodotto diverse in un'unica voce: ogni riga stampata è una voce distinta, anche se i nomi si somigliano o sono adiacenti.
- Alcune righe mostrano un dettaglio di quantità/prezzo unitario, in formati diversi: "Cad 0,99 Pz. 2", "n.6 * 2,00", "2 X 16.00", e simili. Non è mai una voce a sé stante. Per il "prezzo" della voce usa SEMPRE il totale di riga stampato (l'ultimo numero a destra sulla riga del prodotto), MAI il prezzo unitario isolato: se una riga mostra "BLACK ANGUS n.2 * 16,00 ... 32,00", il prezzo della voce è 32,00, non 16,00.
- Solo le righe che iniziano esplicitamente con "SCONTO" (o dicitura equivalente di sconto/riduzione) hanno un prezzo negativo. Tutte le altre voci di prodotto hanno un prezzo positivo, esattamente come stampato.
- "prezzo" è sempre IVA INCLUSA. Se il documento espone il prezzo di riga al NETTO e l'IVA a parte (tipico di fatture e biglietti), metti in "prezzo" la somma netto + IVA di quella riga.

REGOLE SULL'IVA:
- Aliquote italiane valide: 4, 10, 22 (numero). Per una voce "non soggetta a IVA" / "esente" / codice "N2" (regime forfettario/minimi): iva_aliquota = null.
- Se l'IVA totale non è indicata sul documento: iva_totale_stampata = null.
- Non inventare valori: se non leggi con certezza, usa null. Separatore decimale: il punto.

CASI PARTICOLARI:
- BOLLETTA (luce/gas/telefono/acqua): non ha righe di spesa. Lascia "voci": [] e compila solo esercente, data, totale_stampato e, se indicato, iva_totale_stampata (il totale imposte).
- Ricevuta di solo rifornimento carburante da un POS: come la bolletta, "voci": [].
- Se il documento arriva come più immagini: sono pagine dello STESSO documento, trattale come un insieme unico.
- Documento illeggibile o non di spesa: restituisci comunque il JSON con i campi leggibili e null/[] per gli altri.

VERIFICA prima di rispondere: se ci sono voci, la loro somma (sconti col segno) deve tornare con totale_stampato. Se non torna, controlla di non aver fuso righe diverse, creato voci da righe di dettaglio "Cad X Pz. N", o usato il netto invece del lordo.
"""

# Prompt per le voci di una bolletta. È separato da quello di estrazione perché
# le regole sulle bollette, messe nel prompt generale, peggioravano la lettura
# degli altri documenti. Si usa solo dopo che la bolletta è già stata letta.
PROMPT_VOCI_BOLLETTA = """Questo documento è una bolletta (luce, gas, acqua, telefono). Il totale da pagare è {totale} euro. Elenca le voci che compongono questo totale e restituisci ESCLUSIVAMENTE questo oggetto JSON, senza testo prima o dopo, senza markdown, senza commenti:

{"voci": [{"descrizione": "...", "prezzo": 0.00}]}

REGOLE:
- Le voci sono le righe del riepilogo degli importi (di solito "sintesi degli importi fatturati" o "riepilogo"): una voce per riga. Come descrizione copia le parole stampate sulla riga, senza cambiarle; come prezzo l'importo stampato, con il suo segno (una riga può essere negativa).
- Imposte e IVA, se stampate come riga, sono voci a parte.
- La somma delle voci deve dare il totale da pagare. Per questo NON riportare: il totale da pagare; un totale parziale insieme alle righe che somma (o le righe o il loro totale, mai entrambi); la stessa riga due volte; il dettaglio dei consumi (letture, scaglioni, prezzi unitari); importi in lire; importi di altre bollette (bollette precedenti, crediti insoluti).
- Riporta solo importi stampati sul documento: non calcolare e non inventare importi per far tornare la somma.
- Numeri con il punto decimale e senza separatore delle migliaia: 2186.90, non 2.186,90.
- Se non trovi righe che compongono il totale: {"voci": []}.
"""


def _jpeg_da_pil(img: Image.Image, lato_massimo: int) -> bytes:
    """Converte un'immagine in JPEG (bytes), riducendola se il lato più lungo
    supera `lato_massimo`."""
    img = img.convert("RGB")
    larghezza, altezza = img.size
    if max(larghezza, altezza) > lato_massimo:
        fattore = lato_massimo / max(larghezza, altezza)
        img = img.resize((int(larghezza * fattore), int(altezza * fattore)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _immagine_file_a_jpeg(percorso: Path, lato_massimo: int) -> bytes:
    """Carica un file immagine (JPEG/PNG/...) e lo normalizza in JPEG."""
    with Image.open(percorso) as img:
        return _jpeg_da_pil(img, lato_massimo)


def _pdf_estrai_testo(percorso: Path) -> str:
    """Il testo di tutte le pagine di un PDF (vuoto o quasi se è una scansione)."""
    parti = []
    with fitz.open(percorso) as doc:
        for pagina in doc:
            parti.append(pagina.get_text())
    return "\n".join(parti).strip()


def _pdf_pagine_a_jpeg(percorso: Path, lato_massimo: int, max_pagine: int) -> list[bytes]:
    """Converte le prime `max_pagine` pagine di un PDF in JPEG. Si usa solo per
    i PDF scansionati."""
    immagini: list[bytes] = []
    with fitz.open(percorso) as doc:
        if doc.page_count == 0:
            raise ValueError("PDF senza pagine")
        zoom = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)  # 72 DPI è la risoluzione di base di un PDF
        for i in range(min(doc.page_count, max_pagine)):
            pix = doc.load_page(i).get_pixmap(matrix=zoom)
            with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
                immagini.append(_jpeg_da_pil(img, lato_massimo))
    return immagini


def _prepara_documento(percorso: Path) -> tuple[Optional[str], list[bytes]]:
    """Sceglie come dare il documento al modello. Restituisce (testo, immagini):
    uno dei due è pieno. Se c'è il testo si usa la modalità testuale, altrimenti
    si mandano le immagini al modello con vista."""
    if percorso.suffix.lower() == ".pdf":
        testo = _pdf_estrai_testo(percorso)
        if len(testo) >= SOGLIA_TESTO_PDF:
            return testo, []
        return None, _pdf_pagine_a_jpeg(percorso, LATO_MASSIMO_PX, MAX_PAGINE_PDF_SCAN)
    return None, [_immagine_file_a_jpeg(percorso, LATO_MASSIMO_PX)]


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


def _parse_data(valore) -> Optional[date]:
    """Converte la data letta ('AAAA-MM-GG') in un oggetto date. Se manca o non
    è leggibile restituisce None: su un documento reale può succedere, e non è
    un errore da ritentare."""
    if not isinstance(valore, str):
        return None
    testo = valore.strip()
    if testo.lower() in {"", "null", "none", "n/a"}:
        return None
    try:
        return date.fromisoformat(testo)
    except ValueError:
        return None


def _stringa_o_default(valore, default: str = "non leggibile") -> str:
    """Restituisce la stringa ripulita, o `default` se manca. Su alcuni documenti
    il modello legge bene importo e voci ma non il nome dell'esercente: meglio un
    segnaposto che far fallire l'intera estrazione."""
    if isinstance(valore, str) and valore.strip():
        return valore.strip()
    return default


def _normalizza_tipo_documento(valore) -> str:
    """Riporta il tipo di documento letto a uno dei valori ammessi. Se manca
    diventa 'scontrino', se non è riconosciuto diventa 'altro'."""
    validi = {t.value for t in TipoDocumento}
    if not isinstance(valore, str) or not valore.strip():
        return TipoDocumento.SCONTRINO.value
    testo = valore.strip().lower()
    return testo if testo in validi else TipoDocumento.ALTRO.value


def _punteggio_coerenza(documento: DocumentoSenzaCategoria) -> float:
    """Di quanto la somma delle voci si discosta dal totale stampato (più è
    basso, meglio è). Serve a scegliere fra due estrazioni entrambe non valide."""
    return abs(sum(v.prezzo for v in documento.voci) - documento.totale_stampato)


def _voci_bolletta(
    documento: DocumentoSenzaCategoria,
    testo_documento: Optional[str],
    immagini: list[bytes],
    max_tentativi: int,
) -> list[Voce]:
    """Seconda chiamata al modello, solo per le bollette: legge le voci che
    compongono il totale già letto. Le voci si tengono solo se la loro somma
    torna con il totale (lo stesso controllo del Node 2); altrimenti il modello
    riceve l'errore e riprova, e se non ci riesce la bolletta resta senza voci.
    Così questa chiamata non può cambiare il totale né far bocciare la bolletta."""
    prompt = PROMPT_VOCI_BOLLETTA.replace("{totale}", f"{documento.totale_stampato:.2f}")
    if testo_documento is not None:
        messaggio = {
            "role": "user",
            "content": (
                prompt
                + "\n\n=== TESTO INTEGRALE DEL DOCUMENTO (PDF, tutte le pagine) ===\n"
                + testo_documento
            ),
        }
    else:
        messaggio = {"role": "user", "content": prompt, "images": immagini}

    cronologia: list[dict] = [messaggio]
    for _ in range(max_tentativi):
        try:
            risposta = ollama.chat(
                model=MODELLO,
                messages=cronologia,
                think=False,
                options={"temperature": TEMPERATURE},
            )
            contenuto_risposta = risposta["message"]["content"]
            dati = json.loads(_pulisci_output_json(contenuto_risposta))
            voci = [
                Voce(descrizione=v["descrizione"], prezzo=v["prezzo"])
                for v in dati.get("voci", [])
                if v.get("prezzo") is not None
            ]
        except Exception:
            # Chiamata fallita o risposta non utilizzabile: si riprova da capo.
            continue

        if not voci:
            return []

        esito = valida(documento.model_copy(update={"voci": voci}))
        if esito.valido:
            return voci

        cronologia.append({"role": "assistant", "content": contenuto_risposta})
        cronologia.append({
            "role": "user",
            "content": (
                "Le voci non sono coerenti: " + " | ".join(esito.errori) + ". "
                "Ricontrolla: un totale parziale riportato insieme alle righe che somma, "
                "una riga ripetuta, una riga mancante (per esempio imposte e IVA), un importo "
                "che non appartiene a questa bolletta. Restituisci di nuovo SOLO il JSON."
            ),
        })

    return []


def estrai_documento(
    percorso_immagine: Path | str,
    id_documento: str,
    max_tentativi: int = MAX_TENTATIVI_DEFAULT,
) -> tuple[Optional[DocumentoSenzaCategoria], Optional[str]]:
    """Legge il documento e restituisce (documento, errore): se va bene
    (DocumentoSenzaCategoria, None), se fallisce dopo tutti i tentativi
    (None, messaggio d'errore). Non solleva eccezioni per i fallimenti attesi
    (file illeggibile, Ollama che non risponde, JSON non valido), così la
    pipeline non si ferma.

    Se un'estrazione è valida come struttura ma il Node 2 la boccia, e restano
    tentativi, al modello si rimanda l'errore del Node 2 chiedendo di
    ricontrollare le voci. All'ultimo tentativo si restituisce comunque il
    risultato: forzare un altro giro rischierebbe di far alterare al modello
    numeri letti bene, solo per far tornare i conti."""

    try:
        testo_documento, immagini = _prepara_documento(Path(percorso_immagine))
    except FileNotFoundError as e:
        return None, f"File non trovato: {e}"
    except Exception as e:
        return None, (
            f"Impossibile aprire '{percorso_immagine}' come immagine o PDF: "
            f"{type(e).__name__}: {e}"
        )

    # Messaggio iniziale, testo o immagini. `cronologia` è la conversazione: in
    # caso di nuovo tentativo vi si aggiungono la risposta precedente e la correzione.
    if testo_documento is not None:
        messaggio = {
            "role": "user",
            "content": (
                PROMPT_ESTRAZIONE
                + "\n\n=== TESTO INTEGRALE DEL DOCUMENTO (PDF, tutte le pagine) ===\n"
                + testo_documento
            ),
        }
        modalita = "testo"
    else:
        messaggio = {
            "role": "user",
            "content": PROMPT_ESTRAZIONE,
            "images": immagini,
        }
        modalita = "vision"

    cronologia: list[dict] = [messaggio]
    ultimo_errore: Optional[str] = None
    migliore: Optional[DocumentoSenzaCategoria] = None
    punteggio_migliore: float = float("inf")

    for tentativo in range(1, max_tentativi + 1):
        try:
            risposta = ollama.chat(
                model=MODELLO,
                messages=cronologia,
                think=False,
                options={"temperature": TEMPERATURE},
            )
        except Exception as e:
            ultimo_errore = (
                f"[tentativo {tentativo}/{max_tentativi}] ({modalita}) "
                f"Errore di chiamata a Ollama: {e}"
            )
            continue

        contenuto_risposta = risposta["message"]["content"]
        testo_pulito = _pulisci_output_json(contenuto_risposta)

        try:
            dati = json.loads(testo_pulito)
        except json.JSONDecodeError as e:
            ultimo_errore = (
                f"[tentativo {tentativo}/{max_tentativi}] JSON non valido nella risposta "
                f"del modello: {e}. Output grezzo: {testo_pulito!r}"
            )
            continue

        tipo_doc = _normalizza_tipo_documento(dati.get("tipo_documento"))

        if tipo_doc == TipoDocumento.BOLLETTA.value:
            # In questa prima lettura una bolletta resta senza voci: le sue
            # voci si leggono dopo, con una chiamata dedicata (_voci_bolletta).
            voci_grezze = []
        else:
            # Scarta le righe di dettaglio senza prezzo (es. "Cad 0,99 Pz. 2").
            voci_grezze = [v for v in dati.get("voci", []) if v.get("prezzo") is not None]

        try:
            voci = [
                Voce(
                    descrizione=v["descrizione"],
                    prezzo=v["prezzo"],
                    iva_aliquota=v.get("iva_aliquota"),
                )
                for v in voci_grezze
            ]
            documento = DocumentoSenzaCategoria(
                id=id_documento,
                tipo_documento=tipo_doc,
                esercente=_stringa_o_default(dati.get("esercente")),
                tipo_esercizio=_stringa_o_default(dati.get("tipo_esercizio")),
                data=_parse_data(dati.get("data")),
                voci=voci,
                totale_stampato=dati["totale_stampato"],
                iva_totale_stampata=dati.get("iva_totale_stampata"),
            )
        except (KeyError, ValueError, TypeError) as e:
            ultimo_errore = (
                f"[tentativo {tentativo}/{max_tentativi}] ({modalita}) JSON sintatticamente valido "
                f"ma struttura/campi non conformi a quanto atteso: {e}. Dati ricevuti: {dati!r}"
            )
            continue

        # Nessuna voce e totale zero: il modello non ha letto nulla di utile.
        # Meglio un fallimento esplicito che un documento con totale 0.
        if not documento.voci and not documento.totale_stampato:
            ultimo_errore = (
                f"[tentativo {tentativo}/{max_tentativi}] ({modalita}) Estrazione vuota: "
                f"nessuna voce e totale {documento.totale_stampato}. Il documento non è stato "
                f"letto in modo utile."
            )
            continue

        # Se il Node 2 la approva, è fatta. Altrimenti si riprova con il suo errore
        # (righe fuse, righe mancanti, prezzo unitario al posto del totale di riga).
        esito_node2 = valida(documento)
        if esito_node2.valido:
            if documento.tipo_documento == TipoDocumento.BOLLETTA:
                voci_bolletta = _voci_bolletta(documento, testo_documento, immagini, max_tentativi)
                documento = documento.model_copy(update={"voci": voci_bolletta})
            return documento, None

        # Non valida: si tiene il tentativo migliore, non per forza l'ultimo. Il
        # nuovo tentativo può anche peggiorare (è successo: da pochi centesimi di
        # scarto a centinaia di euro).
        punteggio = _punteggio_coerenza(documento)
        if punteggio < punteggio_migliore:
            migliore, punteggio_migliore = documento, punteggio

        if tentativo == max_tentativi:
            return migliore, None

        cronologia.append({"role": "assistant", "content": contenuto_risposta})
        cronologia.append({
            "role": "user",
            "content": (
                "Il documento estratto non è coerente: " + " | ".join(esito_node2.errori) + ". "
                "Ricontrolla con attenzione le voci — possibili righe fuse insieme, righe mancanti, "
                "o un prezzo unitario riportato al posto del totale di riga — e restituisci di nuovo "
                "SOLO il JSON corretto, stessa struttura di prima."
            ),
        })
        ultimo_errore = (
            f"[tentativo {tentativo}/{max_tentativi}] ({modalita}) Node 2 non valido "
            f"({esito_node2.errori}), riprovo con feedback."
        )
        continue

    return None, ultimo_errore
