from typing import List, Dict, Union, Optional
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader as PDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)

class DocumentLoader:
    """Service for loading and preprocessing documents."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def load_documents(self, 
                      source_path: Union[str, Path],
                      file_type: str = "text") -> List[Dict[str, str]]:
        """
        Load documents from a file or directory.
        
        Args:
            source_path: Path to file or directory
            file_type: Type of files to load ("text", "pdf", "markdown")
            
        Returns:
            List of document dictionaries with text content
        """
        source_path = Path(source_path)
        
        # Select appropriate loader
        if source_path.is_file():
            if file_type == "text":
                loader = TextLoader(str(source_path))
            elif file_type == "pdf":
                loader = PDFLoader(str(source_path))
            elif file_type == "markdown":
                loader = UnstructuredMarkdownLoader(str(source_path))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            documents = loader.load()
            
        elif source_path.is_dir():
            glob_pattern = f"**/*.{file_type}"
            loader = DirectoryLoader(
                str(source_path),
                glob=glob_pattern,
                loader_cls=TextLoader
            )
            documents = loader.load()
            
        else:
            raise ValueError(f"Invalid source path: {source_path}")
            
        # Split documents into chunks
        split_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            split_docs.extend([{"text": chunk, "source": str(doc.metadata.get("source", ""))}
                             for chunk in chunks])
            
        return split_docs
    
    def save_documents(self,
                      documents: List[Dict[str, str]],
                      output_path: Union[str, Path]):
        """
        Save processed documents to JSON file.
        
        Args:
            documents: List of document dictionaries
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
            
    def load_from_json(self, json_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load documents from saved JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            List of document dictionaries
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f) 