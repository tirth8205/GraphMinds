# Knowledge Graph Project

This project is designed to work with knowledge graphs using various Python tools and libraries. The environment for the project is managed using Conda, and this `README.md` will guide you through the steps to set up and use the project.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Knowledge Graph Project** leverages tools such as `langchain`, `pandas`, `numpy`, and more to manipulate and work with knowledge graphs. The project is structured in a way that allows users to quickly set up a development environment and start using the provided tools.

## Features

- **Graph Manipulation**: Easily manipulate knowledge graphs using `networkx` and `pyvis`.
- **Data Handling**: Utilise `pandas` for efficient data management.
- **Text Processing**: The project includes tools for splitting and processing documents.
- **Visualisation**: Visualise data using libraries like `seaborn` and `pyvis`.

## Requirements

Before you begin, ensure you have met the following requirements:
- You have installed [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda).
- You are using a supported platform: Windows, MacOS, or Linux.

## Environment Setup

To make it easy to get started, we provide a Conda environment configuration file (`environment.yml`). Follow the steps below to set up the environment and install the necessary dependencies.

### Prerequisites

Ensure you have Conda installed on your system. You can download and install it from the [official Conda website](https://docs.conda.io/en/latest/miniconda.html).

### Steps to Set Up the Environment

1. **Clone the Repository:**

   First, clone this repository to your local machine.

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create the Conda Environment:**

   Use the provided `environment.yml` file to create the environment. Run the following command in your terminal:

   ```bash
   conda env create -f environment.yml
   ```

   This will create a new Conda environment named `knowledge-graph` with all the required dependencies.

3. **Activate the Environment:**

   After the environment is created, activate it with the following command:

   ```bash
   conda activate knowledge-graph
   ```

4. **Verify the Installation:**

   Once the environment is activated, you can verify the installation of the required packages by running:

   ```bash
   conda list
   ```

   You should see packages like `pandas`, `numpy`, `langchain`, and others listed.

5. **Launch JupyterLab (Optional):**

   If you're planning to work in JupyterLab, you can start it with:

   ```bash
   jupyter lab
   ```

6. **Deactivating the Environment:**

   Once you're done, you can deactivate the Conda environment by running:

   ```bash
   conda deactivate
   ```

### Notes

- If you need to install additional packages, you can do so within the activated environment using `conda install` or `pip install`.
- The Python version is not fixed in the environment file, so the latest compatible version of Python will be installed when the environment is created.

## Usage

Once the environment is set up, you can begin working with the project. Run the script or cells one by one.

Make sure your input files are placed in the correct `input` folder, and the results will be written to the `output` folder.

## Contributing

Contributions are always welcome! Please adhere to the following guidelines:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
```

This `README.md` template covers the overview of your project, setup instructions, and usage guidelines. Feel free to update the URLs and project-specific information where needed.
