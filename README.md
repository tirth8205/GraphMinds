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

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/tirth8205/GraphMinds.git
cd GraphMinds
```

### 2. Create and activate the Conda environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate graphminds
```

### 3. Verify the installation by checking the installed packages:
```bash
conda list
```

### 4. Setting Up Ollama

The Ollama API provides the necessary interface for interacting with the LLMs (e.g., `mistral-openorca` and `zephyr:7b`). The Zephyr model is a series of fine-tuned versions of the Mistral and Mixtral models that are trained to act as helpful assistants. Zephyr is a 7B parameter model, distributed under the Apache license, available in both instruction-following and text completion variants.

To use the Ollama API:

1. Download and install the Ollama 5.1.4 API from the official [Ollama website](https://ollama.com/download).

2. After installing the API, verify the installation using the following command to check the version:

   ```bash
   ollama version
   ```

3. Once installed, you can download the required models like `mistral-openorca` and `zephyr:7b` using Ollama's built-in commands:

   ```bash
   ollama pull mistral-openorca
   ollama pull zephyr:7b
   ```

4. Ensure that the models are correctly installed and ready for interaction:

   ```bash
   ollama list
   ```

   This will list all the available models that are ready to use with the project.

### 5. Launch JupyterLab (Optional):
If you're planning to work in JupyterLab, you can start it with:
```bash
jupyter lab
```

### 6. Deactivating the Environment:
Once you're done, you can deactivate the Conda environment by running:
```bash
conda deactivate
```

### Notes:
- If you need to install additional packages, you can do so within the activated environment using `conda install` or `pip install`.
- The Python version is not fixed in the environment file, so the latest compatible version of Python will be installed when the environment is created.

## Usage

1. **Set Up the Environment**: After setting up the environment, open the `extract_graph.ipynb` Jupyter notebook and ensure the kernel is set to **Knowledge Graph** (the environment you just created).

2. **Prepare Data**:
   - Place your PDF file in the `input/` folder (PDFs only at the moment).
   - Open the notebook and update the file name in the script:

   ```python
   # Load the PDF document
   loader = PyPDFLoader("input/#FileName")  # Replace #FileName with the name of your PDF file
   documents = loader.load()  # Load the content of the PDF
   ```

3. **Run the Notebook**: Execute the cells to process the PDF and generate the knowledge graph. Once the processing is done, the system will generate an interactive HTML file for querying the relationships within the document.

4. **Example Query Script**:
   You can also run a predefined script to query the document:

   ```python
   # Example query
   query = "#ENTER YOUR QUERY HERE"  # Replace with your query
   response = answer_query_with_all_relationships(query, contentreplacedforchunk_dfg, df)

   # Print the response
   print(response)
   ```

5. **Alternatively, Start a Chat**: You can initiate an interactive chat as defined in the last cell of the notebook:

   ```python
   while True:
       # Get the user's query
       query = input("Ask your question: ").strip()
       
       # Check if the user wants to exit
       if query.lower() in ['exit', 'quit']:
           print("Ending the interaction. Goodbye!")
           break
   ```

   The script will continuously prompt for questions until you type `exit` or `quit`.

## System Architecture

1. **Embedding Generation**: Generates sentence embeddings for each relationship in the dataset by combining node, edge, and context data.
2. **Cosine Similarity Matching**: Matches user queries with the relationships in the dataset using cosine similarity.
3. **Graph-Based Inference**: Extracts direct and indirect relationships from the knowledge graph to provide context-rich answers.
4. **LLM-Powered Answer Generation**: Uses an LLM to create natural language answers based on the relationships and context.

## Key Innovation

The core of GraphMinds lies in its ability to leverage knowledge graphs and integrate indirect relationships into its analysis. By combining LLMs with structured data from knowledge graphs, GraphMinds enhances the accuracy of insights generated from unstructured and fragmented information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is developed by **Tirth Kanani** under the supervision of **Prof. Christopher Baber** as part of the MSc program in Human-Computer Interaction at the University of Birmingham. Special thanks to the developers of tools such as Sentence Transformers, NetworkX, and PyVis for their invaluable contributions.

For more detailed insights and background on this project, you can access the full project report [here](https://bham-my.sharepoint.com/personal/txk316_student_bham_ac_uk/_layouts/15/guestaccess.aspx?share=EU-tWsvCYNRAl2pF0RFYdRcBpr_e64yTktVMrVpeUS4NNg&e=Xe8N7w).

[GitHub Repository](https://github.com/tirth8205/GraphMinds.git)
