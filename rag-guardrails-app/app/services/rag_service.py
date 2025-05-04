from typing import List, Dict, Optional, Any
import os
from pathlib import Path
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import litellm

from app.services.document_loader import DocumentLoader
from app.services.document_indexer import DocumentIndexer
from app.models.ChatOpenRouter import ChatOpenRouter

load_dotenv()

class RAGService:
    """Main service for RAG operations combining document processing, indexing, and LLM."""
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG service.
        
        Args:
            config_path: Path to NeMo Guardrails config
            model_name: Name of the embedding model
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        # Initialize components
        self.loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.indexer = DocumentIndexer(model_name=model_name)
        
        # Initialize LLM components
        self._init_llm(config_path)
        
        # Initialize document store
        self.documents: List[Dict[str, str]] = []
        
    def _init_llm(self, config_path: Optional[str] = None):
        """Initialize LLM components."""
        # Set OpenRouter API key for evaluation
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY")
        
        # Load NeMo Guardrails config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "config.yml"
            )
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        config = RailsConfig.from_path(config_path)
        
        # Get model configuration from config
        model_config = config.models[0]  # Get the main model config
        
        # Initialize OpenRouter with config values
        self.llm_provider = ChatOpenRouter(
            model=model_config.model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=model_config.parameters.get("temperature", 0.7),
            max_tokens=model_config.parameters.get("max_tokens", 1024),
            top_p=model_config.parameters.get("top_p", 0.95)
        )
        
        # Initialize NeMo Guardrails
        self.rails = LLMRails(
            config,
            llm=self.llm_provider,
            verbose=True  # Enable verbose for debugging
        )
        
    def load_and_index_documents(self,
                               source_path: str,
                               file_type: str = "text",
                               save_dir: Optional[str] = "data") -> None:
        """
        Load documents and create search index.
        
        Args:
            source_path: Path to documents
            file_type: Type of documents to load
            save_dir: Directory to save processed files
        """
        # Load and process documents
        self.documents = self.loader.load_documents(source_path, file_type)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save processed documents
            self.loader.save_documents(
                self.documents,
                save_dir / "documents.json"
            )
            
            # Create and save index
            self.indexer.create_index(self.documents)
            self.indexer.save_index(save_dir / "vector_index.faiss")
            
    def load_existing_index(self,
                          documents_path: str,
                          index_path: str) -> None:
        """
        Load existing documents and index.
        
        Args:
            documents_path: Path to saved documents JSON
            index_path: Path to saved FAISS index
        """
        self.documents = self.loader.load_from_json(documents_path)
        self.indexer.load_index(index_path)
        
    @observe(as_type="generation")
    async def query(self,
                   query: str,
                   num_contexts: int = 2) -> Dict[str, Any]:
        """
        Process a query using the RAG pipeline.
        
        Args:
            query: User query
            num_contexts: Number of context passages to retrieve
            
        Returns:
            Dictionary containing generated response and retrieved contexts
        """
        try:
            # Retrieve relevant contexts
            _, indices = self.indexer.search(query, k=num_contexts)
            retrieved_docs = [self.documents[i] for i in indices[0]]
            contexts = [doc['text'] for doc in retrieved_docs]
            
            # Format the query and context
            formatted_query = {
                "role": "user",
                "content": query
            }
            
            if contexts:
                # Format context as bullet points for clarity
                context_text = "\n".join(f"• {ctx}" for ctx in contexts)
                formatted_query["context"] = context_text
            
            print(f"Sending query to model: {formatted_query}")
            
            # Update model parameters for this query
            self.llm_provider.temperature = 0.1
            self.llm_provider.max_tokens = 512
            
            # Generate response with guardrails
            response = await self.rails.generate_async(
                messages=[formatted_query]
            )
            
            print(f"Received response from model: {response}")
            
            # Handle empty or invalid responses
            if not response or (isinstance(response, dict) and (not response.get('content') or response.get('content').isspace())):
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            elif isinstance(response, dict):
                response = response.get('content', "I apologize, but I couldn't generate a proper response. Please try rephrasing your question.")
            
            return {
                "answer": response,
                "contexts": contexts
            }
            
        except Exception as e:
            print(f"Error in RAG query: {str(e)}")
            raise Exception(f"Error generating response: {str(e)}")

class RAGServiceNoGuardrails:
    """RAG service without Nemo Guardrails, uses ChatOpenRouter directly."""
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.indexer = DocumentIndexer(model_name=model_name)
        self.llm_provider = ChatOpenRouter(
            model=os.environ.get("OPENROUTER_MODEL", "opengvlab/internvl3-2b:free"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0.1,
            max_tokens=512,
            top_p=0.95
        )
        self.documents: List[Dict[str, str]] = []

    def load_and_index_documents(self, source_path: str, file_type: str = "text", save_dir: Optional[str] = "data") -> None:
        self.documents = self.loader.load_documents(source_path, file_type)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.loader.save_documents(self.documents, save_dir / "documents.json")
            self.indexer.create_index(self.documents)
            self.indexer.save_index(save_dir / "vector_index.faiss")

    def load_existing_index(self, documents_path: str, index_path: str) -> None:
        self.documents = self.loader.load_from_json(documents_path)
        self.indexer.load_index(index_path)

    async def query_no_guardrails(self, query: str, num_contexts: int = 2) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        try:
            _, indices = self.indexer.search(query, k=num_contexts)
            retrieved_docs = [self.documents[i] for i in indices[0]]
            contexts = [doc['text'] for doc in retrieved_docs]
            context_text = "\n".join(f"• {ctx}" for ctx in contexts) if contexts else ""
            prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
            logger.info(f"[RAGServiceNoGuardrails] Prompt: {prompt}")
            # Tracing can be added here if needed
            answer = await self.llm_provider.agenerate_text(prompt)
            logger.info(f"[RAGServiceNoGuardrails] Answer: {answer}")
            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            logger.error(f"Error in RAGServiceNoGuardrails query: {str(e)}")
            raise Exception(f"Error generating response: {str(e)}") 