# Assistente conversazionale per la finanza personale

Progetto di tesi triennale (Informatica, Università degli Studi di Perugia).
Relatore: Prof. Francesco Santini.

Si carica in chat la foto o il PDF di un documento di spesa (scontrino,
ricevuta, fattura, bolletta, biglietto). Il sistema legge i dati, controlla che
i numeri tornino, classifica la spesa e la salva in un registro. Poi si possono
fare domande in italiano sulle proprie spese, per esempio
«quanto ho speso per ogni categoria?».

Tutto gira in locale, con un modello di linguaggio eseguito da Ollama: nessun
documento lascia il computer e non serve nessun servizio a pagamento.

## Come funziona

```
documento (foto / PDF)
   │
   ▼
1. Estrazione      un modello che legge immagini ricava esercente, data, voci, totale
2. Validazione     codice Python, senza modello: la somma delle voci fa il totale?
3. Classificazione un modello di testo assegna una categoria (Alimentari, Trasporti, ...)
   │
   ▼
registro SQLite  ──►  domande in italiano  ──►  risposta in chat
```

- Il passo 2 è deliberatamente senza modello: un controllo sui numeri deve dare
  sempre lo stesso esito sullo stesso documento.
- Le domande in italiano vengono tradotte in una query SQL di sola lettura, che
  passa due controlli di sicurezza prima di essere eseguita.

## Come si avvia

Servono Python 3 (sviluppato con la 3.12) e [Ollama](https://ollama.com).

1. Installare il modello (una volta sola):
   ```
   ollama pull qwen3.5:9b
   ```
2. Installare le librerie (una volta sola):
   ```
   pip install -r requirements.txt
   ```
3. Avviare la chat con un doppio clic su `avvia_chat.bat`, oppure da terminale:
   ```
   chainlit run codice/chat.py
   ```
   La chat si apre nel browser all'indirizzo http://localhost:8000.

Dopo la prima elaborazione il registro si crea da solo nella cartella del
progetto (`registro_spese.db`).

Per provare la chat senza scrivere nel registro, mettere a `False` la costante
`SALVA_NEL_REGISTRO` all'inizio di `codice/chat.py`: i documenti vengono
elaborati e mostrati, ma non salvati.

## Come si usa

- **Un documento:** allegare la foto o il PDF. Compaiono esercente, categoria,
  il motivo della categoria, il totale, l'esito del controllo dei numeri e le
  voci. Se i conti non tornano, la chat chiede se salvare comunque.
- **Una domanda:** scrivere solo testo, per esempio «quanto ho speso per ogni
  categoria?».
- **Il riepilogo:** scrivere `riepilogo` per il grafico della spesa per categoria.
- **Un solo documento da terminale:** `python codice/graph.py percorso/del/documento.pdf`.

## Contenuto del repository

| Cartella o file | Che cosa contiene |
|---|---|
| `codice/chat.py` | l'interfaccia di chat |
| `codice/node1_vision.py`, `node2_validator.py`, `node3_classifier.py` | i tre passi di elaborazione |
| `codice/graph.py`, `pipeline_state.py`, `schema.py` | il collegamento dei passi (LangGraph) e le strutture dati |
| `codice/persistenza.py`, `schema_db.sql` | il registro SQLite |
| `codice/agente_interrogazione.py` | le domande in italiano |
| `requirements.txt` | le librerie Python |
| `avvia_chat.bat` | l'avvio con un doppio clic (Windows) |

## Dati

Il sistema è stato provato su 75 documenti di spesa reali. Sono documenti
privati e non fanno parte di questo repository, così come il registro che ne
deriva e gli strumenti di misura che li usano.
