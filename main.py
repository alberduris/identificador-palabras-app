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


# Get the API key from .env and set it 
openai.api_key = st.secrets['OPENAI_API_KEY']

class IdentificadorPalabrasModel(BaseModel):
    palabra: str
    razonamiento: str
    output: bool

def parse_output(initial_response, parser, model, max_retries=3):
    try:
        return json.loads(initial_response.choices[0].message.content)
    except:
        print(
            "No JSON object could be decoded, trying OutputFixingParser...")
        # Instantiate the OutputFixingParser
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(
            openai_api_key=openai.api_key, temperature=0, model_name=model))

        # Loop max_retries times while the response is not valid JSON
        to_fix_response = initial_response.choices[0].message.content
        for i in range(max_retries):
            fixed_response = fixing_parser.parse(
                to_fix_response)  # Get the fixed response (class)
            print(f"Retrying {i+1}/{max_retries}...")
            try:
                return fixed_response.dict()
            except:
                # If still not valid JSON, keep trying looping over response
                print(f"{i+1}/{max_retries}: Another one bites the dust")
                to_fix_response = fixed_response

        print(
            "No JSON object could be decoded, returning as string")
        return initial_response.choices[0].message.content


def getStructuredResponse(system_input: PromptValue, user_input: PromptValue, parser: PydanticOutputParser):

    print("Getting structured response...")

    # Call the model
    structured_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[  # NOTE: HARDCODED GPT-4 for structured response
        {"role": "system", "content": system_input.to_string()},
        {"role": "user", "content": user_input.to_string()}
    ],
        temperature=0,)

    # Process the response
    fixed_response = parse_output(
        structured_response, parser, "gpt-3.5-turbo")  # NOTE: HARDCODED GPT-4 for structured response

    return fixed_response


# Title
st.title("Identificador de palabras")
st.header("Testing suite")

# Palabra Text input
st.markdown("Introduce las palabras a identificar. Si quieres introducir varias, pon una palabra por línea.")
palabras = st.text_area("Palabras", placeholder="TECH")

# Texto Descriptivo text input
st.markdown("Introduce el texto descriptivo.")
texto_descriptivo = st.text_area(
    "Texto descriptivo", placeholder="Aparatos e instalaciones de alumbrado...")

# Botón para ejecutar el modelo
if st.button("Identificar"):

    # Get each word from the "palabras" text area and put them in a list
    palabras = palabras.split("\n")

    
    # SYSTEM INPUT
    # Prepare prompt
    parser = PydanticOutputParser(pydantic_object=IdentificadorPalabrasModel)
    sys_prompt = PromptTemplate(
        template="""Eres un analizador lingüístico y tu tarea principal es decidir si una palabra dada se relaciona directamente con un fragmento de texto.

        Dada una Palabra y un TextoDescriptivo, debes identificar si la Palabra describe o se relacione directamente con el TextoDescriptivo

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

        Tarea 1: Razona si la Palabra está relacionada semánticamente con alguna de las palabras o conceptos del TextoDescriptivo. Aplica pensamiento lateral y pensamiento crítico. 
        Tarea 2: Decide si hay o puede haber (true) o no (false) relación semántica entre Palabra y TextoDescriptivo y responde "true" o "false".

        Tarea: Responde únicamente con un JSON válido siguiendo el OUTPUT SCHEMA. 
        """,
        input_variables=[],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )
    sys_input = sys_prompt.format_prompt()

    responses = []
    with st.spinner("Identificando palabras..."):
        for palabra in palabras:
            # USER INPUT

            # Prepare prompt
            user_prompt = PromptTemplate(
                template="""Palabra: '{palabra}'. TextoDescriptivo: '{texto_descriptivo}'.""",
                input_variables=["palabra", "texto_descriptivo"],
            )
            user_input = user_prompt.format_prompt(
                palabra=palabra.lower(), texto_descriptivo=texto_descriptivo.lower())

            # Call openai chat endpoint
            response = getStructuredResponse(sys_input, user_input, parser)
            responses.append(response)

            with st.expander(f"{response['palabra'].capitalize()}"):
                st.info(f"**Razonamiento**: {response['razonamiento']}")
                if response['output']:
                    st.success("Output: True")
                else:
                    st.error("Output: False")
    
    # Display the responses pretty with 
    # for response in responses:
    #     st.json(response)
