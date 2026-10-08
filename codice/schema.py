"""
Schema dei dati del progetto (Pydantic).

Un documento di spesa (scontrino, ricevuta, fattura, bolletta, biglietto) ha
sempre gli stessi campi. Cambia solo la categoria, che esiste soltanto dopo il
Node 3. Per questo ci sono due classi:

  - DocumentoSenzaCategoria: il documento letto dal Node 1, senza categoria.
  - Documento: lo stesso documento dopo il Node 3, con la categoria obbligatoria.

Si passa dall'una all'altra con DocumentoSenzaCategoria.con_categoria(), l'unico
punto in cui si crea un Documento completo. In questo modo non serve mai una
categoria finta ("non classificato") per i documenti non ancora classificati.

I campi comuni sono scritti una volta sola in _DocumentoCampiComuni.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CategoriaSpesa(str, Enum):
    """Le categorie di spesa assegnate dal Node 3: nove categorie in base al
    tipo di bene o servizio acquistato, più "Altro" per ciò che non rientra
    in nessuna."""

    ALIMENTARI = "Alimentari"
    RISTORAZIONE = "Ristorazione"
    TRASPORTI = "Trasporti"
    CASA_UTENZE = "Casa e utenze"
    SALUTE = "Salute"
    CURA_PERSONA = "Bellezza e cura personale"
    ABBIGLIAMENTO_ACCESSORI = "Abbigliamento e accessori"
    TEMPO_LIBERO = "Tempo libero"
    TECNOLOGIA_SERVIZI_DIGITALI = "Tecnologia e servizi digitali"
    ALTRO = "Altro"


class TipoDocumento(str, Enum):
    """Che tipo di documento è. Lo compila il Node 1 e serve al Node 2 per
    scegliere l'avviso quando un documento arriva senza voci."""

    SCONTRINO = "scontrino"   # documento commerciale: voci con aliquota IVA per riga
    RICEVUTA = "ricevuta"     # ricevuta commerciale tipo scontrino (Amazon, e-commerce)
    FATTURA = "fattura"       # fattura elettronica: imponibile + IVA separati, P.IVA emittente
    BOLLETTA = "bolletta"     # utenza luce/gas/telefono/acqua: le voci sono le righe del riepilogo degli importi
    BIGLIETTO = "biglietto"   # trasporto: bus / volo / traghetto / ingresso
    ALTRO = "altro"


class Voce(BaseModel):
    """Una riga di spesa del documento."""

    descrizione: str
    prezzo: float = Field(
        ...,
        description=(
            "Prezzo della voce, IVA inclusa, in euro. Nelle bollette è l'importo "
            "stampato nel riepilogo, dove imposte e IVA sono una voce a parte."
        ),
    )
    iva_aliquota: Optional[float] = Field(
        default=None,
        description=(
            "Aliquota IVA della voce in percentuale (es. 4, 10, 22). "
            "None quando la voce è legittimamente non soggetta a IVA "
            "(es. fornitore in regime forfettario/minimi)."
        ),
    )


class _DocumentoCampiComuni(BaseModel):
    """Campi comuni a tutti i documenti. Non contiene la categoria, perché
    manca prima del Node 3. Non si usa direttamente (per questo il _ nel nome):
    si usano le due classi sotto."""

    id: str
    tipo_documento: TipoDocumento = Field(
        default=TipoDocumento.SCONTRINO,
        description=(
            "Natura del documento. Sui documenti articolati in voci (scontrino, "
            "ricevuta, fattura, biglietto, bolletta con riepilogo) il Node 2 controlla "
            "la somma delle voci; su quelli senza voci (ricevuta POS carburante, "
            "bolletta senza riepilogo) non c'è nessun controllo numerico, e il tipo "
            "serve a scegliere l'avviso."
        ),
    )
    esercente: str
    tipo_esercizio: str
    data: Optional[date] = Field(
        default=None,
        description=(
            "Data del documento. None quando non è leggibile con certezza: meglio "
            "assente che inventata."
        ),
    )
    voci: list[Voce] = Field(
        default_factory=list,
        description=(
            "Righe di spesa. Per una bolletta sono le righe del riepilogo degli "
            "importi (materia, trasporto, oneri, imposte e IVA). Lista vuota per i "
            "documenti senza righe (ricevute POS carburante): in quel caso contano "
            "solo totale_stampato e iva_totale_stampata."
        ),
    )
    totale_stampato: float = Field(..., description="Totale stampato sul documento, in euro")
    iva_totale_stampata: Optional[float] = Field(
        default=None,
        description=(
            "IVA totale stampata sul documento, se presente. None se non soggetto "
            "a IVA o se l'IVA non è esposta."
        ),
    )


class DocumentoSenzaCategoria(_DocumentoCampiComuni):
    """Il documento dopo il Node 1: tutti i campi letti, ma senza categoria
    (la decide il Node 3). Il Node 2 lavora su questa classe, perché gli
    bastano voci e totale."""

    def con_categoria(self, categoria: CategoriaSpesa) -> "Documento":
        """Crea il Documento completo aggiungendo la categoria. Non modifica
        l'oggetto di partenza: ne costruisce uno nuovo."""
        return Documento(
            **self.model_dump(),
            categoria=categoria,
        )


class Documento(_DocumentoCampiComuni):
    """Il documento completo dopo il Node 3, con la categoria obbligatoria.
    È l'oggetto finale della pipeline."""

    categoria: CategoriaSpesa
