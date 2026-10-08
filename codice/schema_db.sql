-- ============================================================================
--  schema_db.sql — Struttura del registro delle spese (SQLite)
-- ============================================================================
--
--  Due tabelle (`documenti` e `voci`) e due viste (`voci_dettaglio` e
--  `documenti_spesa`). Lo esegue persistenza.py all'inizio. Usa "IF NOT
--  EXISTS", quindi si può rieseguire senza toccare i dati.
--
--  Perché due tabelle: un documento ha un numero variabile di righe (0 per una
--  bolletta, decine per una spesa al supermercato). I documenti stanno in
--  `documenti`, le righe in `voci`, collegate da `voci.documento_id`.
--
--  Chiavi esterne: SQLite le applica solo con  PRAGMA foreign_keys = ON  su ogni
--  connessione (lo fa persistenza.py); senza, ON DELETE CASCADE non scatta.
--
--  Importi: sono REAL (numeri con virgola), come nel resto della pipeline, dove
--  il Node 2 lavora con tolleranze.
-- ============================================================================


-- ----------------------------------------------------------------------------
--  Tabella `documenti` — una riga per ogni documento salvato nel registro
--  (un Documento di schema.py, già classificato dal Node 3).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documenti (

    -- Identificativo del documento: "doc_01".."doc_75" per il dataset, un id
    -- generato dalla chat per i documenti caricati dall'utente.
    id                     TEXT    PRIMARY KEY,

    -- Tipo di documento (enum TipoDocumento), salvato come testo con un CHECK.
    tipo_documento         TEXT    NOT NULL
                                   CHECK (tipo_documento IN (
                                       'scontrino', 'ricevuta', 'fattura',
                                       'bolletta', 'biglietto', 'altro'
                                   )),

    -- Esercente e tipo di esercizio letti dal Node 1. Possono valere "non
    -- leggibile": è il segnaposto che il Node 1 usa quando il campo manca.
    esercente              TEXT    NOT NULL,
    tipo_esercizio         TEXT    NOT NULL,

    -- Data del documento, testo 'AAAA-MM-GG' (così funziona strftime). Può essere
    -- NULL se non è leggibile con certezza. Le domande sul tempo usano questo campo.
    data                   TEXT,

    -- Totale stampato sul documento, in euro (non un ricalcolo).
    totale_stampato        REAL    NOT NULL,

    -- IVA totale stampata, NULL se il documento non è soggetto a IVA o non la riporta.
    iva_totale_stampata    REAL,

    -- Categoria assegnata dal Node 3 (enum CategoriaSpesa). Nel registro entrano
    -- solo documenti già classificati, quindi NOT NULL.
    categoria              TEXT    NOT NULL
                                   CHECK (categoria IN (
                                       'Alimentari', 'Ristorazione', 'Trasporti',
                                       'Casa e utenze', 'Salute', 'Bellezza e cura personale',
                                       'Abbigliamento e accessori', 'Tempo libero',
                                       'Tecnologia e servizi digitali', 'Altro'
                                   )),

    -- Perché il Node 3 ha scelto quella categoria.
    motivazione_categoria  TEXT,

    -- Esito del Node 2: 1 = numeri coerenti, 0 = non coerenti. I documenti bocciati
    -- si salvano comunque (scartarli perderebbe una spesa vera): questa colonna
    -- permette di filtrarli (WHERE node2_valido = 1).
    node2_valido           INTEGER NOT NULL
                                   CHECK (node2_valido IN (0, 1)),

    -- Dettaglio dell'esito: gli errori (vuoto se il documento è valido) e gli
    -- avvisi non bloccanti (es. "validazione parziale: documento senza voci").
    node2_errori           TEXT,
    node2_avvisi           TEXT,

    -- Percorso del file originale (immagine o PDF).
    percorso_file          TEXT,

    -- Quando la riga è stata inserita nel registro (diverso dalla data del documento).
    ingerito_il            TEXT    NOT NULL DEFAULT (datetime('now'))
);


-- ----------------------------------------------------------------------------
--  Tabella `voci` — una riga per ogni riga di spesa (una Voce di schema.py).
--  I documenti senza righe (bollette, ricevute POS) non hanno righe qui.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS voci (

    -- Identificativo della riga, generato da SQLite.
    id            INTEGER PRIMARY KEY,

    -- Documento a cui appartiene la voce. Se si cancella il documento, spariscono
    -- anche le sue voci (ON DELETE CASCADE).
    documento_id  TEXT    NOT NULL
                          REFERENCES documenti(id) ON DELETE CASCADE,

    -- Posizione della voce nel documento (1, 2, 3, ...), in ordine di lettura.
    -- Due voci dello stesso documento non possono avere lo stesso numero (UNIQUE sotto).
    riga          INTEGER NOT NULL,

    -- Descrizione come stampata ("MAXI HAMB.BOV.ADULTO"). Le domande su un prodotto
    -- cercano qui ("quanto ho speso in birre" -> descrizione LIKE '%birr%').
    descrizione   TEXT    NOT NULL,

    -- Prezzo della voce, IVA inclusa, in euro.
    prezzo        REAL    NOT NULL,

    -- Aliquota IVA in percentuale (4, 10, 22), oppure NULL per le esenzioni.
    -- Nessun CHECK: il registro accoglie anche ciò che il Node 2 segnala come
    -- anomalo (es. il dazio doganale al 20%, doc_25); lo segnala node2_valido.
    iva_aliquota  REAL,

    UNIQUE (documento_id, riga)
);


-- ----------------------------------------------------------------------------
--  Vista `voci_dettaglio` — una riga per voce, con già i dati del documento
--  (data, mese, categoria, esercente, esito del Node 2). Si usa per le domande
--  su un prodotto:
--
--    SELECT SUM(prezzo) FROM voci_dettaglio
--    WHERE descrizione LIKE '%birr%' AND mese = '2026-09';
-- ----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS voci_dettaglio AS
SELECT
    v.id                        AS voce_id,
    v.documento_id,
    v.riga,
    v.descrizione,
    v.prezzo,
    v.iva_aliquota,
    d.data,
    strftime('%Y-%m', d.data)   AS mese,
    d.categoria,
    d.esercente,
    d.tipo_documento,
    d.node2_valido
FROM voci v
JOIN documenti d ON d.id = v.documento_id;


-- ----------------------------------------------------------------------------
--  Vista `documenti_spesa` — una riga per documento, con un solo importo di
--  spesa: la somma delle voci se ci sono, altrimenti il totale stampato
--  (bollette). Così le somme non contano due volte i documenti con voci e non
--  dimenticano quelli senza:
--
--    SELECT categoria, SUM(spesa) FROM documenti_spesa
--    WHERE mese = '2026-09' GROUP BY categoria;
-- ----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS documenti_spesa AS
SELECT
    d.id,
    d.data,
    strftime('%Y-%m', d.data)   AS mese,
    d.categoria,
    d.esercente,
    d.tipo_documento,
    d.node2_valido,
    COALESCE(
        (SELECT SUM(v.prezzo) FROM voci v WHERE v.documento_id = d.id),
        d.totale_stampato
    )                           AS spesa
FROM documenti d;
