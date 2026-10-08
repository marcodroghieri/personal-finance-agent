"""
persistenza.py — Il registro delle spese (SQLite).

Salva su disco i documenti già elaborati dalla pipeline, così si possono
interrogare in seguito. Usa solo sqlite3 della libreria standard: non conosce né
immagini né modelli.

Il database ha due tabelle (documenti, voci) e due viste (voci_dettaglio,
documenti_spesa), definite in schema_db.sql. Questo modulo esegue quel file
all'inizio e poi inserisce le righe.

Funzioni:
- inizializza_db(db_path): crea le tabelle se non esistono;
- salva_documento(documento, ...): inserisce un documento e le sue voci in
  un'unica transazione (o tutto o niente).

Come si salvano i dati: gli enum come testo, la data come 'AAAA-MM-GG' (o NULL),
l'esito del Node 2 come bandierina node2_valido (0/1) più il testo di errori e
avvisi. La motivazione della categoria viene dal Node 3 e la passa il chiamante.

Se esiste già un documento con lo stesso id, viene rimpiazzato: rilanciare la
pipeline sullo stesso file non crea duplicati.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from node2_validator import RisultatoValidazione
from schema import Documento

# File di database di default, nella cartella da cui si lancia il programma.
DB_DEFAULT = Path("registro_spese.db")

# Il file SQL con la struttura del database, accanto a questo modulo.
_SCHEMA_SQL = Path(__file__).with_name("schema_db.sql")


def _connetti(db_path: Path | str) -> sqlite3.Connection:
    """Apre la connessione. Attiva le chiavi esterne (SQLite le tiene spente di
    default, e senza non scatterebbe il ON DELETE CASCADE) e permette di leggere
    le colonne per nome."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def inizializza_db(db_path: Path | str = DB_DEFAULT) -> None:
    """Crea tabelle e viste eseguendo schema_db.sql. Si può chiamare
    più volte: se il database esiste già non cambia nulla."""
    sql = _SCHEMA_SQL.read_text(encoding="utf-8")
    conn = _connetti(db_path)
    try:
        with conn:  # transazione: commit se non ci sono eccezioni
            conn.executescript(sql)
    finally:
        conn.close()


def salva_documento(
    documento: Documento,
    *,
    risultato_validazione: RisultatoValidazione,
    motivazione_categoria: str | None = None,
    percorso_file: str | None = None,
    db_path: Path | str = DB_DEFAULT,
) -> None:
    """Inserisce un documento e le sue voci nel registro, in un'unica transazione.
    Se esiste già un documento con lo stesso id, lo rimpiazza.

    documento              il Documento completo prodotto dalla pipeline
    risultato_validazione  l'esito del Node 2 per questo documento
    motivazione_categoria  la spiegazione del Node 3 (facoltativa)
    percorso_file          percorso del file originale (facoltativo)
    db_path                file di database (creato se non esiste)
    """
    inizializza_db(db_path)

    riga_documento = (
        documento.id,
        documento.tipo_documento.value,
        documento.esercente,
        documento.tipo_esercizio,
        documento.data.isoformat() if documento.data is not None else None,
        documento.totale_stampato,
        documento.iva_totale_stampata,
        documento.categoria.value,
        motivazione_categoria,
        1 if risultato_validazione.valido else 0,
        "\n".join(risultato_validazione.errori) or None,
        "\n".join(risultato_validazione.avvisi) or None,
        percorso_file,
    )

    # Le voci, numerate da 1 nell'ordine di lettura del documento.
    righe_voci = [
        (documento.id, numero, voce.descrizione, voce.prezzo, voce.iva_aliquota)
        for numero, voce in enumerate(documento.voci, start=1)
    ]

    conn = _connetti(db_path)
    try:
        with conn:  # commit alla fine, rollback automatico se scatta un'eccezione
            # Se il documento esiste già lo si cancella e lo si riscrive.
            # ON DELETE CASCADE porta via anche le sue voci.
            conn.execute("DELETE FROM documenti WHERE id = ?", (documento.id,))

            conn.execute(
                """
                INSERT INTO documenti (
                    id, tipo_documento, esercente, tipo_esercizio, data,
                    totale_stampato, iva_totale_stampata, categoria,
                    motivazione_categoria, node2_valido, node2_errori,
                    node2_avvisi, percorso_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                riga_documento,
            )
            if righe_voci:
                conn.executemany(
                    """
                    INSERT INTO voci (documento_id, riga, descrizione, prezzo, iva_aliquota)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    righe_voci,
                )
    finally:
        conn.close()
