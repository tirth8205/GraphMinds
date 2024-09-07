# %%
import pandas as pd
import numpy as np
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random

## Input data directory
inputdirectory = Path("./input")
## This is where the output csv files will be written
outputdirectory = Path("./output")


# %%
# Load the PDF document
loader = PyPDFLoader("input/report2.pdf")
documents = loader.load()

# Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

pages = splitter.split_documents(documents)

# Save the chunks to a text file
with open("output/chunks.txt", "w") as file:
    for chunk in pages:
        file.write(chunk.page_content + "\n\n")  # Separate chunks by two newlines

print("Number of chunks = ", len(pages))


# %%
from helpers.df_helpers import documents2Dataframe
df = documents2Dataframe(pages)
print(df.shape)
df.head()

# %%
## This function uses the helpers/prompt function to extract concepts from text
from helpers.df_helpers import df2Graph
from helpers.df_helpers import graph2Df

# %%
import os
import pandas as pd
import numpy as np

# To regenerate the graph with LLM, set this to True
regenerate = True

if regenerate:
    # Extract concepts from the DataFrame using the specified model
    concepts_list = df2Graph(df, model='zephyr:latest')
    # Convert the list of concepts into a DataFrame
    dfg1 = graph2Df(concepts_list)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    
    # Save the generated graph DataFrame to a CSV file
    dfg1.to_csv(os.path.join(outputdirectory, "graph.csv"), sep="|", index=False)
    # Save the original DataFrame chunks to a CSV file
    df.to_csv(os.path.join(outputdirectory, "chunks.csv"), sep="|", index=False)
else:
    # Load the graph DataFrame from an existing CSV file
    dfg1 = pd.read_csv(os.path.join(outputdirectory, "graph.csv"), sep="|")

# Replace empty strings with NaN
dfg1.replace("", np.nan, inplace=True)
# Drop rows with NaN values in 'node_1', 'node_2', or 'edge' columns
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
# Set the initial count value to 4 for each row
dfg1['count'] = 4 

# Print the shape of the cleaned DataFrame
print(dfg1.shape)
# Display the first few rows of the cleaned DataFrame
dfg1.head()


# %%
def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


dfg2 = contextual_proximity(dfg1)
dfg2.tail()
print(dfg1)

# %%
import pandas as pd
import numpy as np

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    
    # Self join with chunk id as the key will create a link between terms occurring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    
    # Drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    
    # Group and count direct edges
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    
    # Drop edges with 1 count (optional, depending on your use case)
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    
    # Create a set of indirect edges based on shared intermediate nodes
    indirect_edges = []
    nodes = dfg2[['node_1', 'node_2']].stack().unique()
    
    for node in nodes:
        # Get all nodes directly connected to the current node
        connected_nodes = pd.concat([
            dfg2[dfg2['node_1'] == node][['node_2', 'chunk_id']],
            dfg2[dfg2['node_2'] == node][['node_1', 'chunk_id']].rename(columns={'node_1': 'node_2'})
        ])
        
        # Create pairs of these connected nodes
        for i in range(len(connected_nodes)):
            for j in range(i + 1, len(connected_nodes)):
                pair = sorted([connected_nodes.iloc[i]['node_2'], connected_nodes.iloc[j]['node_2']])
                chunk_ids = ','.join([connected_nodes.iloc[i]['chunk_id'], connected_nodes.iloc[j]['chunk_id']])
                
                indirect_edges.append((pair[0], pair[1], node, chunk_ids))
    
    # Convert indirect edges into a DataFrame and combine with direct edges
    indirect_df = pd.DataFrame(indirect_edges, columns=["node_1", "node_2", "via_node", "chunk_id"])
    
    # Group by to aggregate chunk_ids and count indirect edges
    indirect_df = (
        indirect_df.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "via_node": "count"})
        .reset_index()
    )
    indirect_df.columns = ["node_1", "node_2", "chunk_id", "count"]
    indirect_df["edge"] = "indirect contextual proximity"
    
    # Merge indirect edges with the direct edges
    final_df = pd.concat([dfg2, indirect_df], ignore_index=True)
    
    # Handle cases where direct and indirect edges might overlap
    final_df = (
        final_df.groupby(["node_1", "node_2", "edge"])
        .agg({"chunk_id": ",".join, "count": "sum"})
        .reset_index()
    )
    
    return final_df

# Usage
dfg2 = contextual_proximity(dfg1)
dfg2.tail()


# %%
print(dfg2)

# %%
dfg = pd.concat([dfg1, dfg2], axis=0)

# %%
# Step 1: Group and aggregate the DataFrame, and assign the result to a new DataFrame
updated_dfg = (
    dfg.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)

# Step 2: Print the resulting DataFrame (optional)
print(updated_dfg)

# Step 3: Save the new DataFrame to a CSV file inside the existing output folder
output_directory = "output"
output_file = os.path.join(output_directory, "updated_dfg_grouped.csv")
updated_dfg.to_csv(output_file, index=False)

print(f"DataFrame saved to {output_file}")


# %%
def remove_duplicate_chunk_ids_and_save(updated_dfg: pd.DataFrame, output_file: str):
    # Iterate over each row in the DataFrame and process the chunk_id column
    updated_dfg['chunk_id'] = updated_dfg['chunk_id'].apply(lambda x: ','.join(sorted(set(x.split(',')))))
    
    # Save the updated DataFrame back to the same file
    updated_dfg.to_csv(output_file, index=False)
    print(f"DataFrame with unique chunk_ids saved to {output_file}")

# Example usage after the grouping and aggregation:
output_directory = "output"
output_file = os.path.join(output_directory, "updated_dfg_grouped.csv")

# Call the function to remove duplicates and save the file
remove_duplicate_chunk_ids_and_save(updated_dfg, output_file)


# %%
import pandas as pd
import os

# Define the path to the chunks.csv file
chunks_file = os.path.join("output", "chunks.csv")

# Load the chunks.csv file into a DataFrame with the correct delimiter
chunks_df = pd.read_csv(chunks_file, delimiter='|')

# Display the first few rows of the DataFrame to confirm it's loaded correctly
print(chunks_df.head())


# %%
import pandas as pd

def replace_chunk_ids_with_content(updated_dfg: pd.DataFrame, chunks_df: pd.DataFrame) -> pd.DataFrame:
    # Create a dictionary to map chunk_id to its corresponding text
    chunk_id_to_content = chunks_df.set_index('chunk_id')['text'].to_dict()
    
    # Function to replace chunk_ids in a row with the corresponding text
    def replace_chunk_id_with_text(chunk_ids):
        # Split the chunk_ids, remove duplicates, and replace each with its content
        unique_chunk_ids = set(chunk_ids.split(','))
        replaced_content = [chunk_id_to_content.get(chunk_id.strip(), f"[Unknown chunk_id: {chunk_id}]") for chunk_id in unique_chunk_ids]
        return ' '.join(replaced_content)
    
    # Create a copy of updated_dfg to store the new content
    contentreplacedforchunk_dfg = updated_dfg.copy()
    
    # Apply the function to the 'chunk_id' column of the new DataFrame
    contentreplacedforchunk_dfg['chunk_id'] = contentreplacedforchunk_dfg['chunk_id'].apply(replace_chunk_id_with_text)
    
    # Rename the 'chunk_id' column to 'text_from_chunk_id'
    contentreplacedforchunk_dfg.rename(columns={'chunk_id': 'text_from_chunk_id'}, inplace=True)
    
    # Return the new DataFrame with content replaced
    return contentreplacedforchunk_dfg

# Example usage:

# Replace chunk_ids with content and get the new DataFrame
contentreplacedforchunk_dfg = replace_chunk_ids_with_content(updated_dfg, chunks_df)

# Save the new DataFrame to a CSV file
output_file = os.path.join("output", "contentreplacedforchunk_dfg.csv")
contentreplacedforchunk_dfg.to_csv(output_file, index=False)

print(f"DataFrame with chunk content saved to {output_file}")


# %%
import pandas as pd
import os
from ollama.client import generate

def update_edge_labels_with_llm(contentreplacedforchunk_dfg: pd.DataFrame, model_name: str) -> pd.DataFrame:
    # Function to interact with the LLM and get the refined edge label
    def get_refined_edge(node_1, node_2, text_from_chunk_id):
        prompt = (
            f"Based on the following context, please provide a short and accurate label "
            f"for the relationship between \"{node_1}\" and \"{node_2}\". "
            f"The label should be concise and meaningful. Do not use any outside knowledge, only use the context provided.\n\n"
            f"Context: \"{text_from_chunk_id}\""
        )
        
        # Call the LLM model using Ollama client
        response, _ = generate(model_name=model_name, prompt=prompt)  # Unpacking the tuple
        refined_edge = response.strip()  # The actual response text
        
        # If the response is empty or not useful, default to "contextual proximity"
        if not refined_edge or refined_edge.lower() in ["", "unknown", "not found"]:
            refined_edge = "contextual proximity"
        
        return refined_edge
    
    # Iterate over each row and update the edge label using the LLM
    for index, row in contentreplacedforchunk_dfg.iterrows():
        node_1 = row['node_1']
        node_2 = row['node_2']
        text_from_chunk_id = row['text_from_chunk_id']
        
        # Get the refined edge label from the LLM
        refined_edge = get_refined_edge(node_1, node_2, text_from_chunk_id)
        
        # Update the edge label in the DataFrame
        contentreplacedforchunk_dfg.at[index, 'edge'] = refined_edge
    
    # Save the updated DataFrame to the CSV file
    output_file = os.path.join("output", "contentreplacedforchunk_dfg.csv")
    contentreplacedforchunk_dfg.to_csv(output_file, index=False)
    
    print(f"Updated DataFrame with refined edge labels saved to {output_file}")
    
    return contentreplacedforchunk_dfg

def check_and_process_file(contentreplacedforchunk_dfg: pd.DataFrame, model_name: str):
    output_file = os.path.join("output", "contentreplacedforchunk_dfg.csv")
    
    if os.path.exists(output_file):
        user_input = input(f"The file {output_file} already exists. Do you want to regenerate it? (yes/no): ").strip().lower()
        
        if user_input == 'no':
            print(f"Using the existing file: {output_file}")
            return pd.read_csv(output_file)
        elif user_input == 'yes':
            print("Regenerating the file...")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            return check_and_process_file(contentreplacedforchunk_dfg, model_name)
    
    # If the file does not exist or the user chose to regenerate it
    return update_edge_labels_with_llm(contentreplacedforchunk_dfg, model_name)

# Example usage:

# Assuming contentreplacedforchunk_dfg is already defined

# Check and process the file
contentreplacedforchunk_dfg = check_and_process_file(contentreplacedforchunk_dfg, model_name="mistral-openorca:latest")


# %%
nodes = pd.concat([contentreplacedforchunk_dfg['node_1'], dfg['node_2']], axis=0).unique()
nodes.shape

# %%
import networkx as nx
G = nx.Graph()

## Add nodes to the graph
for node in nodes:
    G.add_node(
        str(node)
    )

## Add edges to the graph
for index, row in contentreplacedforchunk_dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],  # Use refined_edge to add meaningful labels
        weight=row['count']/4
    )


# %%
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of Communities = ", len(communities))
print(communities)

# %%
import seaborn as sns
palette = "hls"

## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


colors = colors2Community(communities)
colors

# %%
for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]

# %%
from pyvis.network import Network
import os

# Define the output directory and file
output_directory = "./docs"
graph_output_directory = os.path.join(output_directory, "index.html")

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

net = Network(
    notebook=False,
    # bgcolor="#1a1a1a",
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    # font_color="#cccccc",
    filter_menu=False,
)

net.from_nx(G)
# net.repulsion(node_distance=150, spring_length=400)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
# net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
net.show_buttons(filter_=["physics"])

# Corrected show method call without the notebook argument
net.show(graph_output_directory)


# %%
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
from ollama.client import generate

# Load the pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_list):
    """Generate embeddings for a list of texts using a pre-trained SentenceTransformer model."""
    return model.encode(text_list)

def add_embeddings_to_contentreplacedforchunk_dfg(contentreplacedforchunk_dfg: pd.DataFrame) -> pd.DataFrame:
    """Add embeddings to the DataFrame `contentreplacedforchunk_dfg` based on combined text from nodes and edges."""
    # Combine node_1, edge, and node_2 into a single text string
    contentreplacedforchunk_dfg['combined_text'] = contentreplacedforchunk_dfg.apply(lambda row: f"{row['node_1']} {row['edge']} {row['node_2']}", axis=1)
    
    # Generate embeddings for the combined text
    embeddings = generate_embeddings(contentreplacedforchunk_dfg['combined_text'].tolist())
    
    # Store embeddings in the DataFrame
    contentreplacedforchunk_dfg['embedding'] = list(embeddings)
    
    return contentreplacedforchunk_dfg

def generate_final_answer_with_llm(relationships, nodes_entities, context):
    """Generate the final answer using the LLM."""
    prompt = (
        f"Given the following relationships and context, provide a summary answer to the query:\n\n"
        f"Relationships:\n{relationships}\n\n"
        f"Entities Involved:\n{nodes_entities}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    
    full_response, _ = generate("mistral-openorca:latest", prompt)
    return full_response.strip()

def answer_query_with_all_relationships(query: str, contentreplacedforchunk_dfg: pd.DataFrame, df: pd.DataFrame, similarity_threshold=0.3) -> str:
    """Answer a user query by gathering all relevant relationships and generating a final answer with LLM."""
    # Step 1: Generate an embedding for the user query
    query_embedding = generate_embeddings([query])[0]
    
    # Step 2: Compute cosine similarity between the query embedding and embeddings in contentreplacedforchunk_dfg
    contentreplacedforchunk_dfg['similarity'] = contentreplacedforchunk_dfg['embedding'].apply(lambda emb: 1 - cosine(query_embedding, emb))
    
    # Step 3: Filter rows based on a similarity threshold
    relevant_rows = contentreplacedforchunk_dfg[contentreplacedforchunk_dfg['similarity'] >= similarity_threshold]
    
    # Step 4: Gather all relevant relationships and contexts
    relationships = []
    context_list = []
    nodes_entities = set()  # To collect unique nodes and entities
    
    for _, row in relevant_rows.iterrows():
        relationship = f"{row['node_1']} - {row['edge']} - {row['node_2']}"
        relationships.append(relationship)
        context_list.append(get_context_from_chunks(row['text_from_chunk_id'].split(','), df))
        nodes_entities.update([row['node_1'], row['node_2']])
    
    relationships_text = "\n".join(relationships)
    context = " ".join(context_list)
    nodes_entities_text = ", ".join(nodes_entities)
    
    # Step 5: Generate the final answer using the LLM
    final_answer = generate_final_answer_with_llm(relationships_text, nodes_entities_text, context)
    
    # Combine everything for the final output
    answer = (
        f"Final Answer: {final_answer}\n\n"
        f"All Relationships:\n{relationships_text}\n\n"
        f"Entities Involved:\n{nodes_entities_text}\n\n"
        f"Context:\n{context}"
    )
    
    return answer

def get_context_from_chunks(text_from_chunk_ids, df):
    """Retrieve context from the text chunks based on text_from_chunk_ids."""
    relevant_texts = df[df['chunk_id'].isin(text_from_chunk_ids)]['text'].tolist()
    return " ".join(relevant_texts) if relevant_texts else ""

# Ensure `contentreplacedforchunk_dfg` is populated with your graph data, and `df` contains the relevant text chunks.
contentreplacedforchunk_dfg = add_embeddings_to_contentreplacedforchunk_dfg(contentreplacedforchunk_dfg)  # Generate and add embeddings to the DataFrame

# To answer a query:
query = "Tell me about Crimea?"  # Example query
response = answer_query_with_all_relationships(query, contentreplacedforchunk_dfg, df)

# Print the response
print(response)


# %%
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
from ollama.client import generate

# Load the pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_list):
    """Generate embeddings for a list of texts using a pre-trained SentenceTransformer model."""
    return model.encode(text_list)

def add_embeddings_to_dfg(dfg: pd.DataFrame) -> pd.DataFrame:
    """Add embeddings to the DataFrame `dfg` based on combined text from nodes and edges."""
    # Combine node_1, edge, and node_2 into a single text string
    dfg['combined_text'] = dfg.apply(lambda row: f"{row['node_1']} {row['edge']} {row['node_2']}", axis=1)
    
    # Generate embeddings for the combined text
    embeddings = generate_embeddings(dfg['combined_text'].tolist())
    
    # Store embeddings in the DataFrame
    dfg['embedding'] = list(embeddings)
    
    return dfg

def generate_final_answer_with_llm(relationships, nodes_entities, context):
    """Generate the final answer using the LLM."""
    prompt = (
        f"Given the following relationships and context, provide a summary answer to the query:\n\n"
        f"Relationships:\n{relationships}\n\n"
        f"Entities Involved:\n{nodes_entities}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    
    full_response, _ = generate("mistral-openorca:latest", prompt)
    return full_response.strip()

def answer_query_with_all_relationships(query: str, dfg: pd.DataFrame, df: pd.DataFrame, similarity_threshold=0.5) -> str:
    """Answer a user query by gathering all relevant relationships and generating a final answer with LLM."""
    # Step 1: Generate an embedding for the user query
    query_embedding = generate_embeddings([query])[0]
    
    # Step 2: Compute cosine similarity between the query embedding and embeddings in dfg
    dfg['similarity'] = dfg['embedding'].apply(lambda emb: 1 - cosine(query_embedding, emb))
    
    # Step 3: Filter rows based on a similarity threshold
    relevant_rows = dfg[dfg['similarity'] >= similarity_threshold]
    
    # Step 4: Gather all relevant relationships and contexts
    relationships = []
    context_list = []
    nodes_entities = set()  # To collect unique nodes and entities
    
    for _, row in relevant_rows.iterrows():
        relationship = f"{row['node_1']} - {row['edge']} - {row['node_2']}"
        relationships.append(relationship)
        context_list.append(get_context_from_chunks(row['chunk_id'].split(','), df))
        nodes_entities.update([row['node_1'], row['node_2']])
    
    relationships_text = "\n".join(relationships)
    context = " ".join(context_list)
    nodes_entities_text = ", ".join(nodes_entities)
    
    # Step 5: Generate the final answer using the LLM
    final_answer = generate_final_answer_with_llm(relationships_text, nodes_entities_text, context)
    
    # Combine everything for the final output
    answer = (
        f"Final Answer: {final_answer}\n\n"
        f"All Relationships:\n{relationships_text}\n\n"
        f"Entities Involved:\n{nodes_entities_text}\n\n"
        f"Context:\n{context}"
    )
    
    return answer

def get_context_from_chunks(chunk_ids, df):
    """Retrieve context from the text chunks based on chunk IDs."""
    relevant_texts = df[df['chunk_id'].isin(chunk_ids)]['text'].tolist()
    return " ".join(relevant_texts) if relevant_texts else ""

def interactive_pdf_query(dfg: pd.DataFrame, df: pd.DataFrame):
    """Interactive chat function for querying the PDF."""
    print("You can now interact with the PDF. Ask questions about the document.")
    print("Type 'exit' or 'quit' to end the interaction.")
    
    while True:
        # Get the user's query
        query = input("Ask your question: ").strip()
        
        # Check if the user wants to exit
        if query.lower() in ['exit', 'quit']:
            print("Ending the interaction. Goodbye!")
            break
        
        # Answer the query using the function from above
        response = answer_query_with_all_relationships(query, dfg, df)
        
        # Display the response
        print("\n" + response + "\n")

# Ensure `dfg` is populated with your graph data, and `df` contains the relevant text chunks.
dfg = add_embeddings_to_dfg(dfg)  # Generate and add embeddings to the DataFrame

# Start the interactive query session
interactive_pdf_query(dfg, df)



