import json
import openai
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.schema import PromptValue
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate

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

    #print("Getting structured response...")

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
