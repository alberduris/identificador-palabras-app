import os
import streamlit as st
import json
import openai
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.schema import PromptValue
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from utils import parse_output, getStructuredResponse
from schemas import IdentificadorPalabrasModel, TranslationSynonymsModel


def identificarPalabras(palabras: str):
    # Get each word from the "palabras" text area and put them in a list
    palabras = palabras.split("\n")

    # SYSTEM INPUT
    # Prepare prompt
    parser = PydanticOutputParser(pydantic_object=IdentificadorPalabrasModel)
    sys_prompt = PromptTemplate(
        template="""Eres un analizador lingüístico y tu tarea principal es decidir si una palabra dada se relaciona directamente con un fragmento de texto.

        Dada una Palabra y un TextoDescriptivo, debes identificar si la Palabra describe o se relaciona directamente con el TextoDescriptivo.

        Few-shot learning:
        Palabra: Hospital
        TextoDescriptivo: Atención médica a pacientes enfermos y heridos.
        Output: "true"

        Palabra: Hotel
        TextoDescriptivo: Fabricación de muebles de madera
        Output: "false"

        Palabra: Guitarra
        TextoDescriptivo: Enseñanza de instrumentos musicales
        Output: "true"

        {format_instructions}

        Tarea 1: Razona si la Palabra está relacionada con alguna de las palabras o conceptos del TextoDescriptivo. Aplica pensamiento lateral y pensamiento crítico. La palabra será considerará relacionada si indica cuestiones como el género, naturaleza, origen, procedencia, destinación, peso o tamaño, valor o cualidad de algo del TextoDescriptivo.
        Tarea 2: Una vez realizada la Tarea 1, decide si hay o puede haber (true) o no (false) relación entre Palabra y TextoDescriptivo y responde "true" o "false".

        Instrucción: Utiliza el CONTEXTO provisto por el usuario para razonar.

        Tarea: Responde únicamente con un JSON válido siguiendo el OUTPUT SCHEMA. 
        """,
        input_variables=[],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )
    sys_input = sys_prompt.format_prompt()

    responses = []
    progress = st.metric(label="Progreso", value=f"{0}/{len(palabras)}")
    with st.spinner("Identificando palabras..."):
        for i, palabra in enumerate(palabras):

            # Get traduccion and sinonimos
            translationSynonyms = getTranslationSynonyms(
                palabra, texto_descriptivo)
            # with st.expander(f"{palabra}"):
            #     st.json(translationSynonyms)

            # USER INPUT

            # Prepare prompt
            user_prompt = PromptTemplate(
                template="""Palabra: '{translation}{synonyms}'. TextoDescriptivo: '{texto_descriptivo}'.
                ## CONTEXTO ##
                ¿Está la palabra '{translation}{synonyms}' relacionada con alguna de las palabras o conceptos de '{texto_descriptivo}'?
                ¿Describe '{translation}{synonyms}' de alguna manera '{texto_descriptivo}'?
                ¿Es '{translation}{synonyms}' algo relacionado con '{texto_descriptivo}'?
                ##############

                Instrucción: Si la respuesta es sí para alguna de las anteriores preguntas, responde "true". Si la respuesta es no para todas las anteriores preguntas, responde "false".
                """,
                input_variables=["translation",
                                 "synonyms", "texto_descriptivo"],
            )
            if len(translationSynonyms['sinonimos_palabra']) > 0:
                synonyms = ", ".join(translationSynonyms['sinonimos_palabra'])
                synonyms = f" ({synonyms})"
            else:
                synonyms = ""
            user_input = user_prompt.format_prompt(translation=translationSynonyms['traduccion_palabra'],
                                                   synonyms=synonyms, texto_descriptivo=texto_descriptivo)

            # Call openai chat endpoint
            response = getStructuredResponse(sys_input, user_input, parser)
            responses.append(response)

            with st.expander(f"{palabra}"):
                st.info(f"**Razonamiento**: {response['razonamiento']}")
                if response['output']:
                    st.success("Output: True")
                else:
                    st.error("Output: False")

            progress.metric(label="Progreso", value=f"{i+1}/{len(palabras)}")


def getTranslationSynonyms(palabra: str, texto_descriptivo: str) -> TranslationSynonymsModel:
    # SYSTEM INPUT
    # Prepare prompt
    parser = PydanticOutputParser(pydantic_object=TranslationSynonymsModel)
    sys_prompt = PromptTemplate(
        template="""Eres un traductor y especialista lingüístico y tu tarea principal es identificar el idioma de un texto, traducir una palabra dada al idioma detectado y generar sinónimos para la palabra dada.

        Dada una Palabra y un Texto Descriptivo, debes identificar el idioma en el que están escritos el TextoDescriptivo y la Palabra. Si la Palabra no está en el mismo idioma que Texto Descriptivo, debes generar la traducción de Palabra para que coincida con el del Texto Descriptivo. También debes generar sinónimos de Palabra en el idioma traducido.

        {format_instructions}

        Tarea 1: Identifica el idioma en el que está escrito el Texto Descriptivo.
        Tarea 2: Identifica el idioma en el que está escrita la Palabra.
        Tarea 3: Si Palabra está en distinto idioma que el Texto Descriptivo, traduce Palabra al IdiomaDestino. Si están en el mismo idioma, no traduzcas nada. Escribe la Palabra original. 
        Tarea 4: Genera entre 3 y 5 sinónimos de Palabra en IdiomaDestino.

        Instrucción: Si el idioma del Texto Descriptivo y la Palabra es el mismo, no es necesario traducir la Palabra, simplemente escribe la Palabra original tanto en el campo "palabra" como en el campo "traduccion_palabra".
        Instrucción: No generes ninguna explicación. 
        Instrucción: Si las Palabra son siglas, acrónimos, marcas o nombres propios, no es necesario traducirlas. Simplemente escribe la Palabra original tanto en el campo "palabra" como en el campo "traduccion_palabra". 
        Instrucción: Si no es posible traducir la Palabra (e.g., siglas, acrónimos, marcas o nombres propios), escribe la Palabra original tanto en el campo "palabra" como en el campo "traduccion_palabra".
        Instrucción: Si no se pueden generar sinónimos de la Palabra (e.g., siglas, acrónimos, marcas o nombres propios), devuelve un array vacío en el campo "sinonimos_palabra".
        Tarea: Responde únicamente con un JSON válido siguiendo el OUTPUT SCHEMA.
        """,
        input_variables=[],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )
    sys_input = sys_prompt.format_prompt()

    # USER INPUT
    # Prepare prompt
    user_prompt = PromptTemplate(
        template="""
        TextoDescriptivo: '{texto_descriptivo}'.
        Palabra: '{palabra}'.
        """,
        input_variables=["palabra", "texto_descriptivo"],
    )
    user_input = user_prompt.format_prompt(
        palabra=palabra, texto_descriptivo=texto_descriptivo)

    # Call openai chat endpoint
    response = getStructuredResponse(sys_input, user_input, parser)

    return response


# Get the API key from .env and set it
openai.api_key = st.secrets['OPENAI_API_KEY']

# Title
st.title("Identificador de palabras")
st.header("Testing suite")

# Palabra Text input
st.markdown(
    "Introduce las palabras a identificar. Si quieres introducir varias, pon una palabra por línea.")
palabras = st.text_area("Palabras", placeholder="TECH")

# Texto Descriptivo text input
st.markdown("Introduce el texto descriptivo.")
texto_descriptivo = st.text_area(
    "Texto descriptivo", placeholder="Aparatos e instalaciones de alumbrado...")

# Botón para ejecutar el modelo
if st.button("Identificar"):

    identificarPalabras(palabras)
