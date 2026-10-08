@echo off
REM Avvia la chat del progetto con un doppio clic.
REM Si posiziona nella cartella in cui si trova questo file, cosi' il
REM registro (registro_spese.db) viene sempre letto e scritto nello stesso posto.
cd /d "%~dp0"

REM Controlla che il modello sia installato in Ollama; se manca, lo dice e si ferma.
ollama list | findstr /C:"qwen3.5:9b" >nul
if errorlevel 1 (
    echo Il modello qwen3.5:9b non risulta installato, oppure Ollama non e' in esecuzione.
    echo Per installarlo:  ollama pull qwen3.5:9b
    pause
    exit /b 1
)

echo Avvio della chat... si aprira' nel browser all'indirizzo http://localhost:8000
chainlit run codice\chat.py
pause
