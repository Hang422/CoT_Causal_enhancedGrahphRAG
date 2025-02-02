# CoT_Causal_enhancedGrahphRAG

A sophisticated medical question-answering system that combines Chain-of-Thought reasoning with enhanced Graph RAG (Retrieval-Augmented Generation) capabilities, leveraging both causal and knowledge graphs for improved medical domain question answering.

## Features

- **Chain-of-Thought (CoT) Reasoning**: Implements step-by-step reasoning for complex medical questions
- **Dual Graph Architecture**: 
  - Causal Graph for understanding cause-effect relationships
  - Knowledge Graph for domain-specific medical knowledge
- **Enhanced RAG System**: Combines traditional vector-based retrieval with graph-based knowledge retrieval
- **Medical Entity Processing**: Utilizes UMLS (Unified Medical Language System) for entity recognition and linking
- **Flexible Pipeline**: Supports multiple processing stages and ablation studies
- **Comprehensive Logging**: Detailed logging system for LLM interactions and processing steps

## System Architecture

The system consists of several key components:

1. **Query Processing**
   - Entity recognition and CUI (Concept Unique Identifier) mapping
   - Path finding in both causal and knowledge graphs
   - Vector-based similarity search

2. **LLM Integration**
   - Supports multiple LLM models (GPT-3.5-turbo, GPT-4-turbo)
   - Standardized interaction logging
   - Different reasoning strategies (direct answer, chain-of-thought, enhanced information)

3. **Graph Enhancement**
   - Path merging and optimization
   - Coverage analysis
   - Graph-based information enhancement

## Data Sources and Graph Construction

### Database Construction Prerequisites

1. **SemMed Database Access**
   - Obtain license for SemMedDB from the National Library of Medicine
   - Contains over 100 million semantic predications from biomedical literature
   - URL: https://lhncbc.nlm.nih.gov/temp/SemRep_SemMedDB_SKR/SemMed.html
   - For detailed methodology and implementation details, please refer to our paper https://arxiv.org/abs/2501.14892

2. **Graph Construction Process**
   - **Knowledge Graph**:
     - Extract semantic relationships from SemMedDB
     - Clean and deduplicate relationships
     - Filter relevant medical entities and relationships
     - Convert to Neo4j graph format
   
   - **Causal Graph**:
     - Extract causal relationships from knowledge graph
     - Apply causal relationship filters
     - Validate cause-effect relationships
     - Build separate causal graph database

3. **Data Processing Steps**
   - Remove duplicate relationships
   - Normalize entity names
   - Validate relationship types
   - Assign relationship strengths for causal connections
   - Create graph indices for efficient querying

## Setup

### Prerequisites

- Python 3.8+
- Neo4j Database
- OpenAI API access
- Cohere API access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Hang422/CoT_Causal_enhancedGrahphRAG.git
cd CoT_Causal_enhancedGrahphRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the environment:
   - Set up Neo4j database credentials
   - Configure OpenAI and Cohere API keys
   - Set up logging directories

### Configuration

Create a configuration file with the following settings:

```python
config = {
    "openai": {
        "api_key": "your-openai-api-key",
        "model": "gpt-4-turbo",
        "temperature": 0.0
    },
    "cohere": {
        "api_key": "your-cohere-api-key",
        "model": "your-chosen-model"
    },
    "db": {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "your-password",
        "max_connections": 50
    }
}
```

## Usage

### Basic Usage

```python
from pipeline import QuestionProcessor

# Initialize the processor
processor = QuestionProcessor("test_path")

# Process questions from a file
processor.batch_process_file("medmcqa", sample_size=100)

# Compare model performance
compare_models("test_path")
```

### Running Experiments

The system supports various experimental configurations:

```python
# Configure model and run comparison
config.openai['model'] = 'gpt-4-turbo'
compare_models('test-4')

# Run with different models
config.openai['model'] = 'gpt-3.5-turbo'
compare_models('test-3.5')
```

## Evaluation

The system includes comprehensive evaluation tools:

- Accuracy calculation across different processing stages
- Coverage analysis for reasoning chains
- Comparison between enhanced and baseline approaches
- Ablation studies for different components

## Project Structure

```
├── cache/                        # Cached data and intermediate results
├── config/                       # Configuration files
├── logs/                         # System and processing logs
├── neo4j_graphrag/              # Neo4j GraphRAG integration
├── output/                       # Output files and results
├── src/
│   ├── graphrag/
│   │   ├── entity_processor.py   # Medical entity processing
│   │   ├── graph_enhancer.py     # Graph enhancement logic
│   │   └── query_processor.py    # Database query handling
│   ├── llm/
│   │   └── interactor.py         # LLM interaction management
│   └── modules/
│       ├── MedicalQuestion.py    # Question representation
│       └── AccuracyAnalysis.py   # Evaluation tools
├── .DS_Store
├── .env.template                 # Environment variables template
├── .gitignore
└── requirements.txt             # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
Third-Party Components
The neo4j_graphrag directory contains code from Neo4j:

Copyright (c) "Neo4j"
Neo4j Sweden AB [https://neo4j.com]
Licensed under the Apache License, Version 2.0
Original code maintained by Neo4j under Apache 2.0 license

When using this project, please note that while the core project is under MIT license, the Neo4j GraphRAG component follows its original Apache 2.0 license as indicated in its source code headers.
## Citation

If you use this work in your research, please cite:

```bibtex
@misc{CoT_Causal_enhancedGrahphRAG,
  author = {Hang422},
  title = {CoT_Causal_enhancedGrahphRAG},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Hang422/CoT_Causal_enhancedGrahphRAG}
}
```
