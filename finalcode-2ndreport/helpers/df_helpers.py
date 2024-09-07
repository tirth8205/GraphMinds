import uuid  # Importing uuid to generate unique identifiers
import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
from .prompts import extractConcepts  # Importing the extractConcepts function from prompts module
from .prompts import graphPrompt  # Importing the graphPrompt function from prompts module

# Function to convert documents to a DataFrame
def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []  # List to hold the rows of data
    for chunk in documents:  # Iterating over each document chunk
        row = {
            "text": chunk.page_content,  # Storing the text content of the chunk
            **chunk.metadata,  # Adding metadata to the row
            "chunk_id": uuid.uuid4().hex,  # Generating a unique chunk ID
        }
        rows = rows + [row]  # Adding the row to the rows list

    df = pd.DataFrame(rows)  # Creating a DataFrame from the rows list
    return df  # Returning the DataFrame

# Function to extract a list of concepts from a DataFrame
def df2ConceptsList(dataframe: pd.DataFrame) -> list:
    # dataframe.reset_index(inplace=True)  # Resetting the DataFrame index (currently commented out)
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type": "concept"}
        ),  # Applying the extractConcepts function to each row of the DataFrame
        axis=1,  # Applying along rows
    )
    # invalid json results in NaN
    results = results.dropna()  # Dropping rows with NaN values
    results = results.reset_index(drop=True)  # Resetting the index of the results

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()  # Flattening the list of concepts
    return concept_list  # Returning the list of concepts

# Function to convert a list of concepts into a DataFrame
def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)  # Replacing empty spaces with NaN
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])  # Dropping rows where 'entity' is NaN
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()  # Converting entities to lowercase
    )

    return concepts_dataframe  # Returning the DataFrame of concepts

# Function to extract graph-related data from a DataFrame
def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    # dataframe.reset_index(inplace=True)  # Resetting the DataFrame index (currently commented out)
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
    )  # Applying the graphPrompt function to each row of the DataFrame
    # invalid json results in NaN
    results = results.dropna()  # Dropping rows with NaN values
    results = results.reset_index(drop=True)  # Resetting the index of the results

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()  # Flattening the list of graph nodes
    return concept_list  # Returning the list of graph nodes

# Function to convert a list of graph nodes into a DataFrame
def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)  # Replacing empty spaces with NaN
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])  # Dropping rows where 'node_1' or 'node_2' is NaN
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())  # Converting 'node_1' to lowercase
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())  # Converting 'node_2' to lowercase

    return graph_dataframe  # Returning the DataFrame of graph nodes
