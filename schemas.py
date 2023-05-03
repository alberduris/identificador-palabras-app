from pydantic import BaseModel


class IdentificadorPalabrasModel(BaseModel):
    palabra: str
    razonamiento: str
    output: bool

class TranslationSynonymsModel(BaseModel):
    palabra: str
    texto_descriptivo: str
    idioma_palabra: str
    idioma_texto_descriptivo: str
    traduccion_palabra: str
    sinonimos_palabra: str
