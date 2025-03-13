# Text Embedding Utilities

A unified interface for text embedding models with visualization and similarity comparison features.

## Supported Models

- **OpenAI Embeddings**: Via LangChain (requires API key)
  - text-embedding-ada-002
  - text-embedding-3-small
  - text-embedding-3-large

- **Sentence Transformers**: Local embedding models
  - all-MiniLM-L6-v2 (fast, dimension=384)
  - all-mpnet-base-v2 (more accurate, dimension=768)
  - multi-qa-mpnet-base-dot-v1 (good for retrieval)
  - paraphrase-multilingual-mpnet-base-v2 (multilingual)

## Installation

Clone the repo:

```bash
# SSH
git clone git@github.com:SpencerPresley/Embeddings-and-Tokenizers.git

# HTTPS
git clone https://github.com/SpencerPresley/Embeddings-and-Tokenizers.git
```

Enter the directory

```bash
cd Embeddings-and-Tokenizers
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key (for OpenAI embeddings):

```bash
OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Files as is

You can run `pdf_tokenizer.py` and pass in the path to a PDF file as a command line argument.

You can directly run `embeddings.py`

### Basic Custom Usage

```python
from embeddings import get_embedding_model

# Local embeddings (no API key required)
embedder = get_embedding_model("sentence-transformer")

# OpenAI embeddings
# embedder = get_embedding_model("openai", model_name="text-embedding-3-small")

# Embed single text
vector = embedder.embed_text("Hello, world!")

# Batch embedding
texts = ["First document", "Second document", "Related text"]
embeddings = embedder.embed_batch(texts)
```

### Computing Similarities

```python
# Compare two texts
similarity = embedder.similarity(
    "The cat sat on the mat", 
    "A feline was resting on a rug"
)
print(f"Similarity: {similarity:.4f}")

# Generate similarity matrix for multiple texts
sim_matrix = embedder.similarity_matrix(texts)
```

### Visualizing Embeddings

```python
texts = [
    "The cat sat on the mat",
    "The dog played in the yard",
    "Machine learning is fascinating"
]
labels = ["Cat", "Dog", "ML"]

# Basic 2D visualization with PCA
embedder.visualize_embeddings(texts, labels=labels, dimensions=2)

# 3D visualization
embedder.visualize_embeddings(texts, labels=labels, dimensions=3)

# Interactive visualization with Plotly
embedder.visualize_embeddings(texts, labels=labels, dimensions=3, use_plotly=True)

# With automatic clustering
embedder.visualize_embeddings(texts, labels=labels, dimensions=2, auto_cluster=True)
```

## Features

- Unified API across different embedding models
- Similarity calculation (cosine)
- 2D/3D visualizations with PCA
- Interactive visualization with Plotly (if installed)
- Automatic clustering for better visualization of relationships
- Automatic handling of batch operations
- Simple factory function for model selection
- Error handling for missing dependencies

## Example Output

### Static Images

Visualizations are saved as PNG files (`embeddings_2d_ModelName.png` or `embeddings_3d_ModelName.png`)

### Interactive Visualizations

When using Plotly (`use_plotly=True`), interactive HTML files are generated that allow:

- Zooming and rotating (especially useful for 3D)
- Hovering to see labels
- Panning and selecting data points

This requires you have `plotly` installed, assuming you installed `requirements.txt` you'll already have it.

If you did not use `requirements.txt` to install the depedencies and want to use plotly then also `pip install plotly`
