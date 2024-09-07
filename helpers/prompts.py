import sys  # Importing sys to manipulate the system path
from yachalk import chalk  # Importing yachalk for coloured terminal output
sys.path.append("..")  # Adding the parent directory to the system path

import json  # Importing the json library to handle JSON operations
import ollama.client as client  # Importing the client module from ollama for model interaction

# Function to extract concepts from a given text using a model
def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )  # Defining the system prompt to extract concepts and categorise them

    # Generating a response using the model
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    
    try:
        result = json.loads(response)  # Attempting to parse the response as JSON
        result = [dict(item, **metadata) for item in result]  # Adding metadata to each item in the result
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")  # Printing the erroneous response if parsing fails
        result = None  # Setting result to None in case of failure
    
    return result  # Returning the parsed result or None

# Function to generate a graph from a given text using a model
def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"  # Setting a default model if none is provided

    # model_info = client.show(model_name=model)  # Fetching model information (currently commented out)
    # print(chalk.blue(model_info))  # Printing model information in blue (currently commented out)

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the following: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )  # Defining the system prompt to generate a network graph of related concepts

    USER_PROMPT = f"context: ```{input}``` \n\n output: "  # Formatting the user prompt with the input text

    # Generating a response using the model
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    
    try:
        result = json.loads(response)  # Attempting to parse the response as JSON
        result = [dict(item, **metadata) for item in result]  # Adding metadata to each item in the result
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")  # Printing the erroneous response if parsing fails
        result = None  # Setting result to None in case of failure
    
    return result  # Returning the parsed result or None
