# GraphMinds: Leveraging Large Language Models and Knowledge Graphs for Transparent and Efficient AI Systems

[![GitHub](https://img.shields.io/badge/Project-GraphMinds-blue)](https://github.com/tirth8205/GraphMinds.git)

## Abstract

> "I am convinced that the crux of the problem of learning is recognising relationships and being able to use them."  
> — _Christopher Strachey in a letter to Alan Turing, 1954_

GraphMinds addresses the challenges of processing unstructured data and leveraging indirect relationships within sensitive domains. Traditional AI systems, particularly those reliant on cloud infrastructures, present security risks, making local, secure analysis increasingly critical.

Initially developed as a Local Retrieval-Augmented Generation (RAG) system, GraphMinds evolved to tackle the limitations of handling unstructured data by utilising knowledge graphs (KGs) to structure information. This enables the system to infer and represent indirect relationships, enhancing the capabilities of Large Language Models (LLMs) in processing fragmented and complex datasets.

GraphMinds integrates advanced graph-based techniques with LLMs, facilitating the capture of indirect relationships within a knowledge graph. This unique approach improves the system's ability to analyse unstructured data, offering deeper insights and enhanced security for knowledge-intensive tasks. 

Evaluations demonstrate GraphMinds' superiority in analyzing large, unstructured datasets, particularly in fields requiring comprehensive analysis, such as criminal investigations. This innovation underscores its potential as a powerful tool for secure and transparent data analysis.

## Key Features

- **Graph-Based Relationship Mapping**: Extracts direct and indirect relationships between entities from unstructured data and represents them in a knowledge graph.
- **Secure Local AI Processing**: Designed to operate securely in local environments, ensuring data confidentiality without reliance on cloud infrastructure.
- **Embeddings and Similarity Matching**: Uses sentence embeddings to compute similarities between user queries and document relationships.
- **LLM Integration for Comprehensive Analysis**: Integrates with advanced LLMs to generate human-readable answers from relationships and contextual data.

## Technologies Used

- **Sentence Transformers**: For generating sentence embeddings.
- **NetworkX and PyVis**: For graph representation and visualization.
- **SciPy**: For calculating cosine similarity between embeddings.
- **Ollama Client**: For interacting with the LLM.
- **Pandas**: For data manipulation and handling relationships.

Apologies for that! If you are using an `environment.yml` file instead of a `requirements.txt` file to manage dependencies, you'll want to update the `README.md` accordingly. Here's how you can adjust the installation instructions to reflect that you're using `conda` and the `environment.yml` file.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tirth8205/GraphMinds.git
   cd GraphMinds
   ```

2. Create and activate the Conda environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate graphminds
   ```

3. Verify the installation by checking the installed packages:
   ```bash
   conda list
   ```

## Usage

1. **Prepare Data**: Ensure that the relationships and contexts from your PDF or unstructured data are represented in a DataFrame containing nodes, edges, and chunk IDs for mapping entities.

2. **Add Embeddings to Graph Data**:
   Use the `add_embeddings_to_dfg` function to generate embeddings for the combined node and edge data.
   ```python
   dfg = add_embeddings_to_dfg(dfg)
   ```

3. **Start Interactive Querying**:
   Use the `interactive_pdf_query` function to interactively query the document and retrieve relevant relationships.
   ```python
   interactive_pdf_query(dfg, df)
   ```

4. **Example Query**: You can query relationships interactively:
   ```bash
   Ask your question: "What is the relationship between entity X and entity Y?"
   ```

   The system will process the query using embeddings and an LLM to generate a meaningful answer.

## Example Code

Here is a quick example of how to set up and run the query system:

```python
# Step 1: Add embeddings to your graph data
dfg = add_embeddings_to_dfg(dfg)

# Step 2: Start the interactive session for querying
interactive_pdf_query(dfg, df)
```

## System Architecture

1. **Embedding Generation**: Generates sentence embeddings for each relationship in the dataset by combining node, edge, and context data.
2. **Cosine Similarity Matching**: Matches user queries with the relationships in the dataset using cosine similarity.
3. **Graph-Based Inference**: Extracts direct and indirect relationships from the knowledge graph to provide context-rich answers.
4. **LLM-Powered Answer Generation**: Uses an LLM to create natural language answers based on the relationships and context.

## Key Innovation

The core of GraphMinds lies in its ability to leverage knowledge graphs and integrate indirect relationships into its analysis. By combining LLMs with structured data from knowledge graphs, GraphMinds enhances the accuracy of insights generated from unstructured and fragmented information.

## Future Work

- **Automated PDF Parsing**: Integrate tools to automatically parse and extract relationships from PDF files.
- **Support for Additional LLM Models**: Expand support for more advanced LLMs to improve the quality and speed of answer generation.
- **Enhanced Visualization**: Improve real-time graph visualizations during interactive querying.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is developed by **Tirth Kanani** under the supervision of **Prof. Christopher Baber** as part of the MSc program in Human-Computer Interaction at the University of Birmingham. Special thanks to the developers of tools such as Sentence Transformers, NetworkX, and PyVis for their invaluable contributions.

[GitHub Repository](https://github.com/tirth8205/GraphMinds.git)

---
