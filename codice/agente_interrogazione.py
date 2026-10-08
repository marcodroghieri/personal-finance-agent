"""
agente_interrogazione.py — Domande in italiano sul registro delle spese.

Esempio: "quanto ho speso in birre questo mese?". Il modello (Ollama) scrive una
query SQL, il programma la controlla, la esegue sul registro e restituisce le
righe. Non fa parte della pipeline LangGraph: lavora sui dati già salvati.

Il flusso, tutto dentro interroga():
  domanda -> controllo della domanda -> il modello scrive l'SQL
  -> controllo dell'SQL -> esecuzione -> risposta

Se un tentativo fallisce (query rifiutata, oppure errore di SQLite, per esempio
una colonna che non esiste) l'errore viene rimandato al modello in conversazione
e gli si chiede di correggersi, fino a max_tentativi volte.

SICUREZZA. Il registro deve solo essere letto, mai modificato. Non ci si fida
del modello: ci sono due controlli indipendenti, in Python puro.
  (a) _query_e_sicura(): la query deve essere una sola istruzione che inizia con
      SELECT o WITH, senza commenti SQL e senza parole di scrittura (DELETE,
      UPDATE, DROP, ...);
  (b) esegui_query_sicura() apre il database in sola lettura (?mode=ro), quindi
      anche se (a) avesse un buco il file non si può modificare.
Il controllo (a) non è un parser SQL completo; la vera protezione è (b). Il
programma è pensato per un solo utente, il proprietario del registro.

Altri due controlli, anch'essi senza modello: _controlla_domanda() ferma prima
le richieste che non sono domande sulle spese (modifiche, schema del database,
testo senza senso); _query_non_interroga_il_registro() ferma dopo le query che
non leggono il registro.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import ollama

from persistenza import DB_DEFAULT

MODELLO = "qwen3.5:9b"

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_BLOCK_PATTERN = re.compile(
    r"Thinking\.\.\..*?\.\.\.\s*done thinking\.?", re.DOTALL | re.IGNORECASE
)


def _pulisci_residuo_thinking(testo: str) -> str:
    """Toglie il ragionamento che qwen3.5 a volte lascia anche con think=False."""
    testo = _THINK_TAG_PATTERN.sub("", testo)
    testo = _THINKING_BLOCK_PATTERN.sub("", testo)
    return testo.strip()


def _pulisci_output_sql(testo: str) -> str:
    """Ripulisce la risposta del modello: toglie il ragionamento, il blocco
    markdown (```sql ... ```) e un ';' finale, così il controllo di sicurezza può
    dire "una sola istruzione" senza inciampare su un caso innocuo."""
    testo = _pulisci_residuo_thinking(testo).strip()
    if testo.startswith("```"):
        righe = testo.split("\n")
        righe = righe[1:]
        if righe and righe[-1].strip() == "```":
            righe = righe[:-1]
        testo = "\n".join(righe)
    return testo.strip().rstrip(";").strip()


# Descrizione del registro per il prompt: non i CREATE TABLE completi, ma solo
# quello che serve per scrivere query corrette, con degli esempi.
SCHEMA_DESCRIZIONE = """Il registro ha due tabelle e due viste. Usa le VISTE quando possibile: fanno già la giunzione e gli aggregati corretti.

Tabella `documenti` (un documento di spesa: scontrino, ricevuta, fattura, bolletta, biglietto):
- id (testo, es. "doc_63")
- tipo_documento (scontrino/ricevuta/fattura/bolletta/biglietto/altro)
- esercente, tipo_esercizio (testo)
- data (testo 'AAAA-MM-GG', può essere NULL se non leggibile)
- totale_stampato, iva_totale_stampata (numeri, euro)
- categoria (una di: Alimentari, Ristorazione, Trasporti, Casa e utenze, Salute, Bellezza e cura personale, Abbigliamento e accessori, Tempo libero, Tecnologia e servizi digitali, Altro)
- node2_valido (0 o 1: 1 se i numeri del documento sono coerenti)

Tabella `voci` (le singole righe di spesa dentro un documento, es. "BIRRA MORETTI 66CL" 1,20€):
- documento_id, riga, descrizione, prezzo, iva_aliquota

Vista `voci_dettaglio` — per le domande su un PRODOTTO specifico: tutte le colonne di `voci` più `data`, `mese` ('AAAA-MM'), `categoria`, `esercente` già uniti dal documento.
Esempio — "quanto ho speso in birre a settembre 2026?":
SELECT SUM(prezzo) FROM voci_dettaglio WHERE descrizione LIKE '%birr%' AND mese = '2026-09'

Esempio — "è aumentata di prezzo la birra rispetto all'anno scorso?":
SELECT strftime('%Y', data) AS anno, AVG(prezzo) FROM voci_dettaglio WHERE descrizione LIKE '%birr%' GROUP BY anno ORDER BY anno

Vista `documenti_spesa` — per gli AGGREGATI di spesa senza doppio conteggio (un documento conta una volta sola, con voci o senza, es. le bollette): id, data, mese, categoria, esercente, tipo_documento, node2_valido, spesa (il numero unico da sommare).
Esempio — "quanto ho speso in totale a settembre 2026?":
SELECT SUM(spesa) FROM documenti_spesa WHERE mese = '2026-09'
Esempio — "quanto ho speso per categoria questo mese?":
SELECT categoria, SUM(spesa) AS totale FROM documenti_spesa WHERE mese = strftime('%Y-%m', 'now') GROUP BY categoria ORDER BY totale DESC

DATE RELATIVE ("questo mese", "il mese scorso", "quest'anno"): la colonna `mese` è un testo 'AAAA-MM' (es. '2026-08'), NON una data completa. Per confrontarla usa SEMPRE strftime('%Y-%m', ...), mai date('now', ...), che produce 'AAAA-MM-GG' e non è mai uguale a un mese.
Esempio — "quanto ho speso il mese scorso?":
SELECT SUM(spesa) FROM documenti_spesa WHERE mese = strftime('%Y-%m', 'now', '-1 month')
Esempio — "quanto ho speso questo mese?":
SELECT SUM(spesa) FROM documenti_spesa WHERE mese = strftime('%Y-%m', 'now')
Esempio — "quanto ho speso quest'anno?":
SELECT SUM(spesa) FROM documenti_spesa WHERE strftime('%Y', data) = strftime('%Y', 'now')
Se la domanda indica un mese preciso (es. "a settembre 2026"), scrivi il mese letterale come negli esempi sopra.

ATTENZIONE, le due viste NON condividono il nome della colonna dell'importo — non scambiarli:
- `voci_dettaglio` -> la colonna dell'importo si chiama `prezzo` (NON ha una colonna `spesa`)
- `documenti_spesa` -> la colonna dell'importo si chiama `spesa` (NON ha una colonna `prezzo`, e NON ha `descrizione`)
"""

REGOLE_SQL = """REGOLE, senza eccezioni:
- Scrivi UNA sola istruzione SQL, che deve iniziare con SELECT (o WITH se usi una CTE che finisce in un SELECT).
- Mai DELETE, UPDATE, INSERT, DROP, ALTER, CREATE, TRUNCATE, REPLACE, ATTACH, PRAGMA, VACUUM: il registro va SOLO letto.
- Nessun commento SQL (-- oppure /* */), nessun punto e virgola nel mezzo della query.
- Se la domanda nomina un PRODOTTO specifico (es. "quanto ho speso in pane/birra/pesto"), la query DEVE filtrare su `voci_dettaglio.descrizione` con LIKE — non basta filtrare solo per categoria: una domanda su un prodotto NON è una domanda sulla sua categoria, sarebbe una risposta diversa da quella richiesta.
- Per cercare un PRODOTTO usa la RADICE della parola, senza la desinenza, così da trovare insieme singolare e plurale (es. "biscotti" -> LIKE '%biscott%', non '%biscotto%' né '%biscotti%'): sul registro le descrizioni sono scritte nell'una o nell'altra forma.
- Per cercare un ESERCENTE (negozio, azienda) usa LIKE '%parola%' sul nome, MAI l'uguaglianza esatta: lo stesso esercente compare nel registro scritto in modi diversi (con o senza ragione sociale, indirizzo, sigle), e l'uguaglianza ne troverebbe solo una parte.
- Rispondi ESCLUSIVAMENTE con la query SQL, senza markdown, senza spiegazioni prima o dopo."""


def _costruisci_prompt_sql(domanda: str) -> str:
    return f"""Sei un assistente che traduce domande in italiano sulle spese personali di un utente in query SQL di sola lettura su un registro SQLite.

{SCHEMA_DESCRIZIONE}

{REGOLE_SQL}

Domanda dell'utente: {domanda}
"""


# Controllo di sicurezza (a): parole di scrittura che una query non deve contenere.
PAROLE_VIETATE = {
    "delete", "update", "insert", "drop", "alter", "create", "truncate",
    "replace", "attach", "detach", "pragma", "vacuum", "reindex",
    "exec", "execute", "begin", "commit", "rollback",
}


def _query_e_sicura(sql: str) -> tuple[bool, str | None]:
    """Restituisce (True, None) se la query è accettabile, altrimenti
    (False, motivo). Non solleva mai eccezioni: una query malformata è
    semplicemente rifiutata, non fa fallire il chiamante."""
    corpo = sql.strip()
    if not corpo:
        return False, "query vuota"

    if "--" in corpo or "/*" in corpo:
        return False, "la query contiene un commento SQL ('--' o '/*'), non ammesso"

    if ";" in corpo:
        return False, "la query contiene più istruzioni (';' nel mezzo), non ammesso"

    prima_parola = re.match(r"^\s*(\w+)", corpo)
    prima_parola = prima_parola.group(1).upper() if prima_parola else ""
    if prima_parola not in {"SELECT", "WITH"}:
        return False, f"la query deve iniziare con SELECT o WITH, non '{prima_parola}'"

    trovate = {
        parola for parola in PAROLE_VIETATE
        if re.search(rf"\b{parola}\b", corpo, re.IGNORECASE)
    }
    if trovate:
        return False, f"la query contiene parole non ammesse: {', '.join(sorted(trovate))}"

    return True, None


# Controlli prima e dopo il modello. Se la richiesta non è una domanda sulle
# spese, il modello non ha modo di dirlo e improvvisa (ripete la domanda, risponde
# a un'altra, esegue quello che gli si chiede). Si risponde allora con un messaggio
# fisso, senza passare dal modello.
MSG_SOLA_LETTURA = (
    "Il registro è di sola lettura: posso solo rispondere a domande "
    "sulle spese registrate."
)
MSG_FUORI_AMBITO = "Posso rispondere solo a domande sulle spese del registro."
MSG_NON_CAPITO = (
    "Non ho capito la domanda: puoi riformularla indicando un prodotto, "
    "un esercente o un mese?"
)

# Verbi di modifica del registro, a parola intera: «cancelleria» non scatta,
# «cancella» sì.
_VERBI_DI_MODIFICA = re.compile(
    r"\b(elimina|eliminare|cancella|cancellare|modifica|modificare|aggiungi|"
    r"aggiungere|inserisci|inserire|rimuovi|rimuovere|svuota|svuotare|azzera|"
    r"azzerare|drop|delete|update|insert|truncate|alter)\b",
    re.IGNORECASE,
)

# Richieste di far uscire il modello dal suo compito o di vedere come è fatto.
_RICHIESTE_SU_ISTRUZIONI = (
    re.compile(r"\b(ignora|dimentica|scorda)\w*\b.*\b(istruzion\w*|regol\w*|precedent\w*)\b", re.IGNORECASE),
    re.compile(r"\b(schema|struttura|tabelle|colonne)\b.*\b(database|db|registro|dati)\b", re.IGNORECASE),
    re.compile(r"\b(prompt|sqlite_master|sqlite_schema)\b", re.IGNORECASE),
)

_VOCALI = set("aeiouàèéìòù")


def _sembra_una_parola(token: str) -> bool:
    """Euristica larga: una parola ha almeno tre lettere, qualche vocale e non
    ha cinque consonanti di fila. Serve a riconoscere il testo senza senso."""
    if len(token) < 3 or not (set(token) & _VOCALI):
        return False
    if sum(c in _VOCALI for c in token) / len(token) < 0.2:
        return False
    return re.search(r"[^aeiouàèéìòù]{5,}", token) is None


def _controlla_domanda(domanda: str) -> str | None:
    """Controllo prima del modello: restituisce il messaggio da mostrare se la
    domanda va fermata, altrimenti None."""
    testo = (domanda or "").strip().lower()
    token = re.findall(r"[a-zàèéìòù]+", testo)
    if not any(_sembra_una_parola(t) for t in token):
        return MSG_NON_CAPITO
    if _VERBI_DI_MODIFICA.search(testo):
        return MSG_SOLA_LETTURA
    if any(p.search(testo) for p in _RICHIESTE_SU_ISTRUZIONI):
        return MSG_FUORI_AMBITO
    return None


def _query_non_interroga_il_registro(sql: str) -> bool:
    """Controllo dopo il modello: True se la query non legge il registro, cioè
    non ha nessun FROM (per esempio SELECT 'Che ore sono?') o interroga le tabelle
    di sistema di SQLite."""
    return (
        re.search(r"\bfrom\b", sql, re.IGNORECASE) is None
        or re.search(r"\bsqlite_\w+", sql, re.IGNORECASE) is not None
    )


@dataclass
class RisultatoInterrogazione:
    """Esito di una domanda in linguaggio naturale al registro."""

    domanda: str
    sql: str | None = None
    righe: list[dict] | None = None
    risposta: str = ""
    errore: str | None = None  # valorizzato solo in caso di fallimento

    def __str__(self) -> str:
        # Si mostra l'SQL anche in caso di errore, per capire cosa ha scritto il modello.
        intestazione = f"SQL: {self.sql}\n" if self.sql else ""
        if self.errore:
            return f"{intestazione}[ERRORE] {self.errore}"
        return f"{intestazione}Risposta: {self.risposta}"


def esegui_query_sicura(
    sql: str, db_path: Path | str = DB_DEFAULT
) -> tuple[list[dict] | None, str | None]:
    """Esegue la query solo se passa il controllo (a), che ripete anche se
    interroga() l'ha già fatto, e la esegue su una connessione in sola lettura (b)."""
    ok, motivo = _query_e_sicura(sql)
    if not ok:
        return None, f"Query rifiutata dal controllo di sicurezza: {motivo}"

    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            righe = [dict(r) for r in conn.execute(sql).fetchall()]
        finally:
            conn.close()
        return righe, None
    except sqlite3.Error as e:
        return None, f"Errore SQLite in esecuzione: {type(e).__name__}: {e}"


def _formatta_risposta(righe: list[dict]) -> str:
    """Elenca le righe come coppie 'colonna: valore'."""
    if not righe:
        return "Nessun risultato trovato per questa domanda."

    MASSIMO_RIGHE_MOSTRATE = 20
    parti = [
        ", ".join(f"{colonna}: {valore}" for colonna, valore in riga.items())
        for riga in righe[:MASSIMO_RIGHE_MOSTRATE]
    ]
    testo = "\n".join(parti)
    if len(righe) > MASSIMO_RIGHE_MOSTRATE:
        testo += f"\n... e altre {len(righe) - MASSIMO_RIGHE_MOSTRATE} righe (troncato)."
    return testo


def interroga(
    domanda: str,
    db_path: Path | str = DB_DEFAULT,
    max_tentativi: int = 2,
) -> RisultatoInterrogazione:
    """Domanda in italiano -> RisultatoInterrogazione (SQL, righe, risposta).

    Se un tentativo fallisce, l'errore viene rimandato al modello in
    conversazione (cronologia) e gli si chiede di correggere la query. Qui non
    si tiene il "tentativo migliore" come nel Node 1: una query o funziona o non
    dà niente."""
    messaggio = _controlla_domanda(domanda)
    if messaggio:
        return RisultatoInterrogazione(
            domanda=domanda, righe=[{"messaggio": messaggio}], risposta=messaggio
        )

    cronologia: list[dict] = [{"role": "user", "content": _costruisci_prompt_sql(domanda)}]
    ultimo_errore: str | None = None
    ultimo_sql: str | None = None

    for tentativo in range(1, max_tentativi + 1):
        try:
            risposta = ollama.chat(
                model=MODELLO,
                messages=cronologia,
                think=False,
                options={"temperature": 0.1, "num_predict": 400},
            )
        except Exception as e:
            ultimo_errore = f"[tentativo {tentativo}/{max_tentativi}] Errore di chiamata a Ollama: {e}"
            continue

        contenuto_risposta = risposta["message"]["content"]
        sql = _pulisci_output_sql(contenuto_risposta)
        ultimo_sql = sql

        ok, motivo = _query_e_sicura(sql)
        if not ok:
            ultimo_errore = (
                f"[tentativo {tentativo}/{max_tentativi}] Query rifiutata dal controllo di "
                f"sicurezza: {motivo}. Query: {sql!r}"
            )
            if tentativo == max_tentativi:
                break
            cronologia.append({"role": "assistant", "content": contenuto_risposta})
            cronologia.append({
                "role": "user",
                "content": (
                    f"La query non è ammessa: {motivo}. Riscrivi SOLO una query SELECT "
                    "(o WITH) di sola lettura, una sola istruzione, senza commenti, senza "
                    "nessuna delle parole vietate."
                ),
            })
            continue

        if _query_non_interroga_il_registro(sql):
            return RisultatoInterrogazione(
                domanda=domanda, sql=sql,
                righe=[{"messaggio": MSG_FUORI_AMBITO}], risposta=MSG_FUORI_AMBITO,
            )

        righe, errore_esecuzione = esegui_query_sicura(sql, db_path=db_path)
        if errore_esecuzione is None:
            return RisultatoInterrogazione(
                domanda=domanda, sql=sql, righe=righe, risposta=_formatta_risposta(righe)
            )

        ultimo_errore = f"[tentativo {tentativo}/{max_tentativi}] {errore_esecuzione}"
        if tentativo == max_tentativi:
            break
        cronologia.append({"role": "assistant", "content": contenuto_risposta})
        cronologia.append({
            "role": "user",
            "content": (
                f"L'esecuzione di questa query ha dato questo errore: {errore_esecuzione}. "
                "Ricontrola nello schema descritto sopra quali colonne esistono su quale "
                "vista — in particolare 'descrizione' esiste solo su 'voci'/'voci_dettaglio', "
                "'prezzo' solo su 'voci'/'voci_dettaglio', 'spesa' solo su 'documenti_spesa' — "
                "e restituisci di nuovo SOLO la query SQL corretta, mantenendo tutti i filtri "
                "della domanda originale (non toglierne uno solo per far sparire l'errore)."
            ),
        })
        continue

    return RisultatoInterrogazione(domanda=domanda, sql=ultimo_sql, errore=ultimo_errore)
