import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_validator import ExcelDataValidator
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_EXTENSIONS = {
    '.csv': 'csv',
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.txt': 'text',
    '.xlsx': 'excel',
    '.xls': 'excel',
    '.pptx': 'powerpoint',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml'
}

class DocumentProcessor:
    """Enhanced document processing with parallel processing and improved error handling"""
    
    @staticmethod
    def load_csv(file_path: str, encoding: str = 'utf-8') -> List[Document]:
        """Load CSV files with automatic delimiter detection and improved error handling"""
        try:
            # Try to detect delimiter
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
                delimiter = ',' if ',' in first_line else ';' if ';' in first_line else '\t'
            
            # Try LangChain loader first
            loader = CSVLoader(
                file_path=file_path,
                encoding=encoding,
                csv_args={'delimiter': delimiter}
            )
            return loader.load()
        except Exception as e:
            logger.warning(f"LangChain CSV loader failed: {e}, falling back to pandas")
            try:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                return DocumentProcessor._dataframe_to_documents(df, file_path)
            except Exception as e2:
                logger.error(f"Pandas CSV loading failed: {e2}")
                return []

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load PDF files with improved error handling"""
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"PDF loading failed: {e}")
            return []

    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """Load Word documents with improved error handling and formatting"""
        try:
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()
            
            # Her dokümanı daha iyi formatla
            formatted_docs = []
            for doc in raw_docs:
                # Metni temizle ve formatla
                content = doc.page_content.strip()
                
                # Paragrafları ayır ve numaralandır
                paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
                formatted_content = []
                
                for i, para in enumerate(paragraphs, 1):
                    if para:  # Boş paragrafları atla
                        formatted_content.append(f"Paragraf {i}: {para}")
                
                # Metadata'yı genişlet
                metadata = {
                    "source": file_path,
                    "file_type": "docx",
                    "total_paragraphs": len(formatted_content),
                    "file_name": os.path.basename(file_path),
                    "processed_date": datetime.now().isoformat(),
                    "chunk_index": len(formatted_docs)  # Chunk indeksini ekle
                }
                
                # Yeni doküman oluştur
                formatted_doc = Document(
                    page_content="\n".join(formatted_content),
                    metadata=metadata
                )
                formatted_docs.append(formatted_doc)
            
            logger.info(f"Successfully processed DOCX file: {file_path}")
            return formatted_docs
            
        except Exception as e:
            logger.error(f"DOCX loading failed: {e}")
            return []

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Load text files with improved error handling"""
        try:
            loader = TextLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Text loading failed: {e}")
            return []

    @staticmethod
    def load_excel(file_path: str) -> List[Document]:
        """Load Excel files with improved error handling"""
        try:
            loader = UnstructuredExcelLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Excel loading failed: {e}")
            return []

    @staticmethod
    def load_powerpoint(file_path: str) -> List[Document]:
        """Load PowerPoint files with improved error handling"""
        try:
            loader = UnstructuredPowerPointLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"PowerPoint loading failed: {e}")
            return []

    @staticmethod
    def load_json(file_path: str) -> List[Document]:
        """Load JSON files with improved error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return DocumentProcessor._json_to_documents(data, file_path)
        except Exception as e:
            logger.error(f"JSON loading failed: {e}")
            return []

    @staticmethod
    def load_yaml(file_path: str) -> List[Document]:
        """Load YAML files with improved error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return DocumentProcessor._json_to_documents(data, file_path)
        except Exception as e:
            logger.error(f"YAML loading failed: {e}")
            return []

    @staticmethod
    def _dataframe_to_documents(df: pd.DataFrame, source: str) -> List[Document]:
        """Convert DataFrame to Document objects with improved metadata"""
        documents = []
        for i, row in df.iterrows():
            # Create a structured content string
            content_parts = []
            for col, val in row.items():
                if pd.notna(val):  # Skip NaN values
                    content_parts.append(f"{col}: {val}")
            
            content = "\n".join(content_parts)
            metadata = {
                "source": source,
                "row": i,
                "columns": list(df.columns),
                "file_type": "csv",
                "total_rows": len(df),
                "total_columns": len(df.columns)
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    @staticmethod
    def _json_to_documents(data: Union[Dict, List], source: str, prefix: str = "") -> List[Document]:
        """Convert JSON/YAML data to Document objects with improved metadata"""
        documents = []
        
        def process_value(value, key_path):
            if isinstance(value, (dict, list)):
                for k, v in value.items() if isinstance(value, dict) else enumerate(value):
                    new_path = f"{key_path}.{k}" if key_path else str(k)
                    process_value(v, new_path)
            else:
                content = f"{key_path}: {value}"
                metadata = {
                    "source": source,
                    "key_path": key_path,
                    "file_type": "json" if source.endswith('.json') else "yaml",
                    "value_type": type(value).__name__
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        process_value(data, prefix)
        return documents

    @staticmethod
    def process_file(file_path: str) -> List[Document]:
        """Process a single file with enhanced support for DOCX"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            # DOCX dosyaları için özel işleme
            if file_extension == '.docx':
                return DocumentProcessor.load_docx(file_path)
            
            if file_extension not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            file_type = SUPPORTED_EXTENSIONS[file_extension]
            loader_method = getattr(DocumentProcessor, f"load_{file_type}", None)
            
            if loader_method:
                docs = loader_method(file_path)
                # Her dokümana chunk indeksi ekle
                for i, doc in enumerate(docs):
                    doc.metadata["chunk_index"] = i
                return docs
            return []
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

    @staticmethod
    def _process_excel(file_path: str) -> List[Document]:
        """Process Excel file with data validation"""
        try:
            # Excel verilerini doğrula ve temizle
            validator = ExcelDataValidator()
            df = pd.read_excel(file_path)
            
            # Veri temizleme ve doğrulama
            cleaned_df, errors = validator.clean_data(df)
            validation_results = validator.validate_data(cleaned_df)
            
            # Kalite raporu oluştur
            quality_report = validator.generate_quality_report(cleaned_df, validation_results)
            
            # Her sayfayı ayrı bir doküman olarak işle
            documents = []
            
            # Excel içeriğini metin olarak ekle
            excel_content = f"Excel Dosyası: {file_path}\n\n"
            excel_content += f"Veri Kalite Raporu:\n{quality_report}\n\n"
            excel_content += f"Veri Özeti:\n{cleaned_df.describe().to_string()}\n\n"
            excel_content += f"İlk 5 Satır:\n{cleaned_df.head().to_string()}\n"
            
            # Metadata oluştur
            metadata = {
                "source": file_path,
                "file_type": "excel",
                "total_rows": len(cleaned_df),
                "total_columns": len(cleaned_df.columns),
                "validation_status": "valid" if validation_results['is_valid'] else "invalid",
                "has_errors": bool(validation_results['errors']),
                "has_warnings": bool(validation_results['warnings'])
            }
            
            # Doküman oluştur
            doc = Document(
                page_content=excel_content,
                metadata=metadata
            )
            documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            return []

    @staticmethod
    def process_files_parallel(file_paths: List[str], max_workers: int = 4) -> List[Document]:
        """Process multiple files in parallel for improved performance"""
        all_documents = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(DocumentProcessor.process_file, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    logger.info(f"Successfully processed {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        return all_documents 