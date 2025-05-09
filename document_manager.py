import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib
import json
import re
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class CorporateDocumentManager:
    """Profesyonel şirket doküman yönetim sistemi"""
    
    def __init__(
        self,
        data_dir: str = "./data",
        metadata_dir: str = "./metadata",
        vector_store_dir: str = "./vector_store",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.metadata_dir = Path(metadata_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processor = DocumentProcessor()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.metadata = self._load_metadata()
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # Initialize vector store if it exists
        if (self.vector_store_dir / "chroma.sqlite3").exists():
            try:
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_store_dir),
                    embedding_function=self.embeddings
                )
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self.vector_store = None
        
        # Gelişmiş text splitter konfigürasyonu
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Öncelik sırasına göre ayırıcılar
            separators=[
                # Önce paragrafları ayır
                "\n\n",
                # Sonra cümleleri ayır (Türkçe cümle sonu işaretleri)
                ". ", "! ", "? ", "... ", "… ",
                # Sonra alt paragrafları ayır
                "\n",
                # En son kelimeleri ayır
                " ", ""
            ],
            # Cümle bölünmesini önlemek için
            keep_separator=True
        )
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file"""
        metadata_file = self.metadata_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save metadata to file"""
        try:
            with open(self.metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""

    def _is_file_updated(self, file_path: Path) -> bool:
        """Check if file has been updated"""
        file_hash = self._calculate_file_hash(file_path)
        return self.metadata.get(str(file_path), {}).get("hash") != file_hash

    def _preprocess_text(self, text: str) -> str:
        """Metni chunk'lara bölmeden önce ön işleme"""
        # Gereksiz boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Cümle sonlarını standardize et
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r'…', '...', text)
        
        # Paragrafları düzenle
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _process_document(self, file_path: str) -> List[Document]:
        """Tek bir dokümanı işle"""
        try:
            docs = self.processor.process_file(file_path)
            
            processed_docs = []
            for doc in docs:
                # Metni ön işle
                processed_text = self._preprocess_text(doc.page_content)
                
                # Yeni Document oluştur
                processed_doc = Document(
                    page_content=processed_text,
                    metadata=doc.metadata.copy()
                )
                
                # Metadata'yı genişlet
                processed_doc.metadata.update({
                    'processed_date': datetime.now().isoformat(),
                    'document_id': f"{os.path.basename(file_path)}_{len(docs)}",
                    'chunk_index': docs.index(doc),
                    'original_length': len(doc.page_content),
                    'processed_length': len(processed_text)
                })
                
                processed_docs.append(processed_doc)
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return []
    
    def process_documents(self, force_update: bool = False) -> bool:
        """Process all documents in the data directory"""
        try:
            if not self.data_dir.exists():
                logger.warning(f"Data directory {self.data_dir} does not exist")
                return False

            documents = []
            for file_path in self.data_dir.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in [".docx", ".pdf", ".txt"]:
                    if force_update or self._is_file_updated(file_path):
                        try:
                            # Process document
                            processed_docs = self.processor.process_file(str(file_path))
                            if processed_docs:  # Check if we got any documents
                                # Add metadata to each document in the list
                                for doc in processed_docs:
                                    doc.metadata.update({
                                        "source": str(file_path),
                                        "file_type": file_path.suffix.lower()[1:],
                                        "processed_date": datetime.now().isoformat(),
                                        "hash": self._calculate_file_hash(file_path)
                                    })
                                    documents.append(doc)
                                
                                # Update metadata
                                self.metadata[str(file_path)] = {
                                    "hash": self._calculate_file_hash(file_path),
                                    "processed_date": datetime.now().isoformat(),
                                    "file_type": file_path.suffix.lower()[1:],
                                    "document_count": len(processed_docs)
                                }
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                            continue

            if documents:
                # Create or update vector store
                if self.vector_store is None:
                    self.vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=str(self.vector_store_dir)
                    )
                else:
                    self.vector_store.add_documents(documents)
                
                # Save metadata
                self._save_metadata()
                logger.info(f"Processed {len(documents)} documents successfully")
                return True
            
            return False

        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document statistics"""
        try:
            stats = {
                "total_documents": len(self.metadata),
                "file_types": {},
                "last_updated": None,
                "vector_store_initialized": self.vector_store is not None
            }
            
            for file_info in self.metadata.values():
                file_type = file_info.get("file_type", "unknown")
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                
                processed_date = file_info.get("processed_date")
                if processed_date:
                    processed_date = datetime.fromisoformat(processed_date)
                    if stats["last_updated"] is None or processed_date > stats["last_updated"]:
                        stats["last_updated"] = processed_date
            
            if stats["last_updated"]:
                stats["last_updated"] = stats["last_updated"].isoformat()
            
            return stats
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    def search_documents(self, query: str, k: int = 8) -> List[Document]:
        """Search documents using MMR"""
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []
            
            return self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=min(k * 2, 20)  # Fetch more documents for better diversity
            )
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def _initialize_vector_store(self):
        """Initialize vector store with improved error handling"""
        try:
            self.vector_store = Chroma(
                persist_directory=os.path.join(self.vector_store_dir, "documents"),
                embedding_function=OpenAIEmbeddings()
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            # Get collection stats
            collection = self.vector_store._collection
            stats = {
                "Total Documents": collection.count(),
                "Vector Store Location": os.path.join(self.vector_store_dir, "documents"),
                "Chunk Size": self.chunk_size,
                "Chunk Overlap": self.chunk_overlap
            }
            
            # Get document types
            doc_types = {}
            for doc in collection.get()["metadatas"]:
                doc_type = doc.get("file_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            stats["Document Types"] = doc_types
            
            return stats
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"error": str(e)} 