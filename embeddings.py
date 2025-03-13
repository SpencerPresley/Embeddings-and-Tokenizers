"""
Text Embedding Utilities

This module provides a unified interface for different embedding models, including:
- OpenAI embeddings via LangChain
- HuggingFace embeddings via sentence-transformers
- Default all-MiniLM embeddings (local)

Each embedding type is wrapped in a consistent interface for easy swapping.
"""

from dotenv import load_dotenv
import numpy as np
from typing import List, Optional
import os

# For visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Optional plotly import for interactive 3D visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

# Import embedding providers
try:
    from langchain_openai import OpenAIEmbeddings

    HAVE_LANGCHAIN_OPENAI = True
except ImportError:
    HAVE_LANGCHAIN_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer

    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False


class BaseEmbedding:
    """Base class for all embedding models"""

    def __init__(self):
        self.name = "Base Embedding"
        self.dimension = 0
        self.is_ready = False

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        raise NotImplementedError("Subclasses must implement this method")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings"""
        raise NotImplementedError("Subclasses must implement this method")

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text embeddings"""
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Generate a similarity matrix for a list of texts"""
        embeddings = self.embed_batch(texts)
        return cosine_similarity(embeddings)

    def visualize_embeddings(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        dim_reduction: str = "pca",
        dimensions: int = 2,
        use_plotly: bool = False,
        auto_cluster: bool = False,
        n_clusters: int = 0,
    ) -> None:
        """
        Visualize embeddings in 2D or 3D after dimensionality reduction

        Args:
            texts: List of texts to embed and visualize
            labels: Optional labels for each text (uses text snippets if not provided)
            dim_reduction: Dimensionality reduction technique ('pca' only for now)
            dimensions: Target dimensions (2 or 3)
            use_plotly: Whether to use Plotly for interactive visualization (requires plotly package)
            auto_cluster: Whether to automatically color points by cluster
            n_clusters: Number of clusters to use (if auto_cluster is True). If 0, will estimate
        """
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3")

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        # Apply dimensionality reduction
        if dim_reduction.lower() == "pca":
            reducer = PCA(n_components=dimensions)
            reduced_embeddings = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unsupported dimensionality reduction: {dim_reduction}")

        # Create labels if not provided
        if labels is None:
            # Create short snippets from the texts
            labels = [t[:20] + "..." if len(t) > 20 else t for t in texts]

        # Make sure we don't have more labels than texts
        if len(labels) > len(texts):
            labels = labels[: len(texts)]

        # Cluster points if requested
        if auto_cluster:
            # Determine number of clusters if not specified
            if n_clusters <= 0:
                # Heuristic: estimate number of clusters based on data size
                n_clusters = min(max(2, len(texts) // 5), 8)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(reduced_embeddings)
        else:
            clusters = None

        # Use Plotly for interactive 3D visualization if requested
        if use_plotly:
            if not HAVE_PLOTLY:
                print("Warning: Plotly not installed. Falling back to matplotlib.")
                print("Install plotly with: pip install plotly")
            else:
                # Use only labels for hover text to keep it clean
                if dimensions == 3:
                    # Create 3D scatter plot
                    if auto_cluster:
                        fig = px.scatter_3d(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            z=reduced_embeddings[:, 2],
                            color=clusters,
                            text=labels,
                            hover_name=labels,
                            title=f"3D Text Embeddings - {self.name}",
                            labels={"color": "Cluster"},
                        )
                    else:
                        fig = px.scatter_3d(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            z=reduced_embeddings[:, 2],
                            text=labels,
                            hover_name=labels,
                            title=f"3D Text Embeddings - {self.name}",
                        )

                    # Add text labels
                    fig.update_traces(
                        textposition="top center",
                        marker=dict(size=6, opacity=0.8),
                    )

                    # Improve layout and set a reasonable size
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="Component 1",
                            yaxis_title="Component 2",
                            zaxis_title="Component 3",
                        ),
                        margin=dict(l=0, r=0, b=0, t=40),
                        width=1200,
                        height=800,
                    )

                    # Save interactive HTML
                    fig.write_html(f"embeddings_3d_{self.name.replace(' ', '_')}.html")

                    # Show plot
                    fig.show()
                    return

                else:  # 2D plot with Plotly
                    if auto_cluster:
                        fig = px.scatter(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            color=clusters,
                            text=labels,
                            hover_name=labels,
                            title=f"2D Text Embeddings - {self.name}",
                            labels={"color": "Cluster"},
                        )
                    else:
                        fig = px.scatter(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            text=labels,
                            hover_name=labels,
                            title=f"2D Text Embeddings - {self.name}",
                        )

                    fig.update_traces(
                        textposition="top center",
                        marker=dict(size=10, opacity=0.8),
                    )

                    fig.update_layout(
                        xaxis_title="Component 1",
                        yaxis_title="Component 2",
                        margin=dict(l=50, r=50, b=50, t=50),
                        width=1200,
                        height=800,
                    )

                    # Save interactive HTML
                    fig.write_html(f"embeddings_2d_{self.name.replace(' ', '_')}.html")

                    # Show plot
                    fig.show()
                    return

        # Fall back to matplotlib visualization with smaller figure sizes
        plt.figure(figsize=(8, 6) if dimensions == 2 else (9, 7))

        if dimensions == 2:
            # Create 2D scatter plot with different colors for clusters if requested
            if auto_cluster:
                scatter = plt.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    c=clusters,
                    cmap="viridis",
                    alpha=0.8,
                    s=60,  # Smaller point size
                )
                plt.colorbar(scatter, label="Cluster")
            else:
                plt.scatter(
                    reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.8, s=60
                )

            # Add labels with improved positioning
            for i, label in enumerate(labels):
                # Add slight offset to avoid labels overlapping with points
                plt.annotate(
                    label,
                    (reduced_embeddings[i, 0] + 0.02, reduced_embeddings[i, 1] + 0.02),
                    fontsize=9,  # Smaller font
                    alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.3),
                )

            plt.xlabel("Component 1", fontsize=10)
            plt.ylabel("Component 2", fontsize=10)
            plt.grid(alpha=0.3)

        else:  # 3D
            # Create new 3D figure with smaller size
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")

            # Create 3D scatter plot with different colors for clusters if requested
            if auto_cluster:
                scatter = ax.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    reduced_embeddings[:, 2],
                    c=clusters,
                    cmap="viridis",
                    s=60,  # Smaller point size
                    alpha=0.8,
                )
                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
                cbar.set_label("Cluster", fontsize=10)
            else:
                ax.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    reduced_embeddings[:, 2],
                    s=60,  # Smaller point size
                    alpha=0.8,
                )

            # Add labels with improved positioning and background
            for i, label in enumerate(labels):
                ax.text(
                    reduced_embeddings[i, 0],
                    reduced_embeddings[i, 1],
                    reduced_embeddings[i, 2],
                    label,
                    fontsize=8,  # Smaller font
                    alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )

            ax.set_xlabel("Component 1", fontsize=10)
            ax.set_ylabel("Component 2", fontsize=10)
            ax.set_zlabel("Component 3", fontsize=10)
            ax.grid(alpha=0.3)

            # Improve default view angle
            ax.view_init(elev=30, azim=45)

        plt.title(f"Text Embeddings - {self.name}", fontsize=12)
        plt.tight_layout()

        # Save static image
        plt.savefig(
            f"embeddings_{dimensions}d_{self.name.replace(' ', '_')}.png", dpi=300
        )
        plt.show()


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI embeddings via LangChain.

    Requires:
    - langchain-openai package
    - OPENAI_API_KEY environment variable set

    Models available: "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
    """

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 8):
        """
        Initialize OpenAI embeddings

        Args:
            model_name: OpenAI embedding model to use
            batch_size: Number of texts to embed in a single batch
        """
        super().__init__()
        self.name = f"OpenAI ({model_name})"

        if not HAVE_LANGCHAIN_OPENAI:
            raise ImportError(
                "OpenAI embeddings require the langchain-openai package. "
                "Install with: pip install langchain-openai"
            )

        # Check for API key
        load_dotenv()
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        except:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it with: export OPENAI_API_KEY='your-api-key'"
                "or add a .env file to the root of the project with the key."
            )

        # Initialize the embedding model
        try:
            self.client = OpenAIEmbeddings(model=model_name, api_key=openai_api_key)

            # Set embedding dimension based on model
            if model_name == "text-embedding-ada-002":
                self.dimension = 1536
            elif model_name == "text-embedding-3-small":
                self.dimension = 1536
            elif model_name == "text-embedding-3-large":
                self.dimension = 3072
            else:
                # Unknown model, we'll have to get the dimension from a test embedding
                test_emb = self.client.embed_query("test")
                self.dimension = len(test_emb)

            self.batch_size = batch_size
            self.is_ready = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI embeddings: {e}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        if not self.is_ready:
            raise RuntimeError("Embedding model is not initialized")

        embedding = self.client.embed_query(text)
        return np.array(embedding)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings"""
        if not self.is_ready:
            raise RuntimeError("Embedding model is not initialized")

        embeddings = self.client.embed_documents(texts)
        return np.array(embeddings)


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Local embedding model using sentence-transformers.

    Popular models:
    - "all-MiniLM-L6-v2" (fast, dimension=384)
    - "all-mpnet-base-v2" (more accurate, dimension=768)
    - "multi-qa-mpnet-base-dot-v1" (good for retrieval, dimension=768)
    - "paraphrase-multilingual-mpnet-base-v2" (multilingual, dimension=768)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model

        Args:
            model_name: Name of the sentence-transformer model
        """
        super().__init__()
        self.name = f"SentenceTransformer ({model_name})"

        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "SentenceTransformer embeddings require the sentence-transformers package. "
                "Install with: pip install sentence-transformers"
            )

        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.is_ready = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        if not self.is_ready:
            raise RuntimeError("Embedding model is not initialized")

        embedding = self.model.encode(text)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings"""
        if not self.is_ready:
            raise RuntimeError("Embedding model is not initialized")

        embeddings = self.model.encode(texts)
        return embeddings


def get_embedding_model(
    model_type: str = "sentence-transformer", **kwargs
) -> BaseEmbedding:
    """
    Factory function to get an embedding model

    Args:
        model_type: Type of embedding model ('openai', 'sentence-transformer')
        **kwargs: Additional arguments for the specific model

    Returns:
        An initialized embedding model
    """
    if model_type.lower() == "openai":
        return OpenAIEmbedding(**kwargs)
    elif model_type.lower() in ["sentence-transformer", "st", "sentence_transformer"]:
        return SentenceTransformerEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Sample texts for demonstration
    texts = [
        "The cat sat on the mat",
        "The dog played in the yard",
        "A feline was resting on a rug",
        "The canine was playing outside",
        "Machine learning is fascinating",
        "Artificial intelligence is transforming industries",
        "Apple designs and produces consumer electronics and software",
        "Oranges are rich in vitamin C and antioxidants",
        "Microsoft develops Windows and Office software products",
        "Google provides search, cloud computing, and advertising services",
        "Technology continues to advance at an accelerating pace",
        "Software development requires logical thinking and creativity",
        "iPhone revolutionized the smartphone industry with its touchscreen",
        "Android is an open-source mobile operating system",
        "MacOS offers a Unix-based operating system with a polished interface",
        "Windows is the most widely used desktop operating system",
    ]

    print("Available embedding types:")
    print(" - OpenAI (requires API key + langchain-openai package)")
    print(" - SentenceTransformer (local, requires sentence-transformers package)")
    print("\nDemonstrating SentenceTransformer embeddings (all-MiniLM-L6-v2):")

    # Create a sentence transformer embedding model
    try:
        embedder = get_embedding_model("sentence-transformer")
        print(f"Model: {embedder.name}")
        print(f"Embedding dimension: {embedder.dimension}")

        # Get embeddings for sample texts
        print("\nGenerating embeddings for sample texts...")
        embeddings = embedder.embed_batch(texts)
        print(f"Shape of embeddings: {embeddings.shape}")

        # Calculate similarities
        print("\nSimilarity between 'cat sat on mat' and 'feline on rug':")
        sim = embedder.similarity(texts[0], texts[2])
        print(f"Cosine similarity: {sim:.4f}")

        print("\nSimilarity matrix for all texts:")
        sim_matrix = embedder.similarity_matrix(texts)
        # Print in a readable format
        for i, row in enumerate(sim_matrix):
            print(f"Text {i+1}: {' '.join([f'{val:.2f}' for val in row])}")

        # Visualize embeddings
        print("\nVisualizing embeddings in 2D (close plot window to continue)...")
        labels = [
            "Cat",
            "Dog",
            "Feline",
            "Canine",
            "ML",
            "AI",
            "Apple",
            "Orange",
            "Microsoft",
            "Google",
            "Technology",
            "Software",
            "iPhone",
            "Android",
            "MacOS",
            "Windows",
        ]
        embedder.visualize_embeddings(
            texts, labels=labels, dimensions=3, use_plotly=True
        )

    except ImportError as e:
        print(f"Error: {e}")
        print("Install required packages to run this example")

    # Demonstrate OpenAI embeddings (requires API key)
    embedder = get_embedding_model("openai", model_name="text-embedding-3-small")
    embeddings = embedder.embed_batch(texts)
    sim = embedder.similarity(texts[0], texts[2])
    embedder.visualize_embeddings(texts, labels=labels, dimensions=3, use_plotly=True)
    print(f"OpenAI similarity: {sim:.4f}")
