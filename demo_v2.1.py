import streamlit as st
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from datetime import datetime
from document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from document_manager import CorporateDocumentManager
from reasoning_chain import MultiHopReasoningChain
import pandas as pd
from pyngrok import ngrok
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
METADATA_DIR = "./metadata"
VECTOR_STORE_DIR = "./vector_store"
MAX_RETRIES = 3
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MAX_TOKEN_LIMIT = 3000
STREAMLIT_PORT = 8501  # Streamlit'in varsayılan portu

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

class EnhancedConversationMemory:
    """Gelişmiş konuşma hafızası yönetimi"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages = []
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        self.token_count = 0  # Token sayacı eklendi
        
    def add_message(self, role: str, content: str):
        """Yeni mesaj ekle ve vektör deposunu güncelle"""
        self.messages.append({"role": role, "content": content})
        
        # Maksimum mesaj sayısını kontrol et
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Vektör deposunu güncelle
        if role == "assistant":
            self._update_vector_store(content)
            
        # Token sayısını güncelle (yaklaşık olarak)
        self.token_count += len(content.split()) // 0.75  # Yaklaşık token sayısı
    
    def _update_vector_store(self, content: str):
        """Vektör deposunu güncelle"""
        try:
            # Yeni mesajı vektör deposuna ekle
            if self.vector_store is None:
                self.vector_store = Chroma(
                    collection_name="conversation_history",
                    embedding_function=self.embeddings
                )
            
            # Mesajı Netpak bağlamında işle
            metadata = {
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "source": "conversation",
                "company": "Netpak"
            }
            
            self.vector_store.add_texts(
                texts=[content],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logging.error(f"Vektör deposu güncelleme hatası: {str(e)}")
    
    def get_relevant_messages(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """Sorgu ile ilgili mesajları getir"""
        try:
            if self.vector_store is None:
                return []
            
            # Netpak bağlamında benzer mesajları ara
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter={"company": "Netpak"}
            )
            
            return [{"role": "assistant", "content": doc.page_content} for doc in results]
            
        except Exception as e:
            logging.error(f"İlgili mesajları getirme hatası: {str(e)}")
            return []
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Konuşma geçmişini getir"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def get_streamlit_messages(self) -> List[Dict[str, str]]:
        """Streamlit için mesajları getir"""
        return self.get_chat_history()
    
    def clear(self):
        """Hafızayı temizle"""
        self.messages = []
        self.token_count = 0  # Token sayacını sıfırla
        if self.vector_store:
            self.vector_store.delete_collection()
            
    def get_statistics(self) -> Dict[str, Any]:
        """Hafıza istatistiklerini getir"""
        return {
            "Total Messages": len(self.messages),
            "Current Token Count": int(self.token_count),
            "Memory Location": os.path.join(VECTOR_STORE_DIR, "conversation_memory")
        }

def verify_netpak_documents(documents: List[Document]) -> bool:
    """Dokümanların Netpak'a ait olup olmadığını doğrula"""
    try:
        # Netpak'a özgü anahtar kelimeler
        netpak_keywords = [
            "Netpak",
            "POL-01",  # Politika dokümanlarının numaralandırma formatı
            "Sosyal Uygunluk Politikası",
            "İş Sağlığı ve Güvenliği Politikası",
            "Çevre Politikası"
        ]
        
        # Her dokümanı kontrol et
        for doc in documents:
            content = doc.page_content.lower()
            metadata = doc.metadata
            
            # Dosya adından kontrol
            if "source" in metadata:
                source = metadata["source"].lower()
                if any(keyword.lower() in source for keyword in netpak_keywords):
                    continue
            
            # İçerikten kontrol
            if any(keyword.lower() in content for keyword in netpak_keywords):
                continue
            
            # Eğer doküman Netpak'a ait değilse
            logging.warning(f"Non-Netpak document detected: {metadata.get('source', 'unknown')}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error verifying Netpak documents: {str(e)}")
        return False

def create_advanced_rag_chain():
    """Gelişmiş RAG zinciri oluştur"""
    try:
        # Model ve tokenizer'ı yükle
        model = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=4000
        )
        
        # Doküman yükleme ve işleme
        loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader
        )
        documents = loader.load()
        
        # Dokümanları işle
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Her chunk'a Netpak metadata'sı ekle
        for split in splits:
            split.metadata.update({
                "company": "Netpak",
                "document_type": "policy" if "POL-01" in str(split.metadata.get("source", "")) else "data"
            })
        
        # Embedding modeli
        embeddings = OpenAIEmbeddings()
        
        # Vector store oluştur
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_DIR
        )
        
        # Retriever'ı yapılandır
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 2,
                "filter": {"company": "Netpak"}
            }
        )
        
        # Multi-hop reasoning chain oluştur
        reasoning_chain = MultiHopReasoningChain(
            model=model,
            retriever=retriever
        )
        
        # Dil tespiti için prompt
        language_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Soru hangi dilde sorulmuş, tespit et.
            Sadece 'tr' veya 'en' olarak cevap ver.
            
            ÖNEMLİ: Sadece dil tespiti yap, başka bir işlem yapma."""),
            ("human", "{question}")
        ])
        
        # Dil tespiti zinciri
        language_detection_chain = language_detection_prompt | model | StrOutputParser()
        
        # Ana zincir
        def process_question(question: str, chat_history: List[Dict[str, str]] = None) -> str:
            try:
                # Dil tespiti yap
                language = language_detection_chain.invoke({"question": question})
                st.write("🔍 Dil tespiti:", language)
                
                # Konuşma geçmişini sınırla (son 3 mesaj gibi)
                limited_chat_history = (chat_history or [])[-3:]
                st.write("💬 Kullanılan konuşma geçmişi:", len(limited_chat_history), "mesaj")
                
                # Soruyu işle
                st.write("🤔 Soru işleniyor...")
                response = reasoning_chain.invoke(
                    question=question,
                    chat_history=limited_chat_history
                )
                
                # Log dosyasından son bilgileri oku ve göster
                with open('app.log', 'r', encoding='utf-8') as f:
                    logs = f.readlines()
                    last_logs = logs[-20:]  # Son 20 log satırını al
                    
                    st.write("📊 Model İstatistikleri:")
                    for log in last_logs:
                        if "Retrieved context length" in log:
                            st.write("📚 Alınan bağlam uzunluğu:", log.split(" - ")[-1].strip())
                        elif "Created subquestions" in log:
                            st.write("❓ Oluşturulan alt sorular:", log.split(" - ")[-1].strip())
                        elif "Context length" in log:
                            st.write("📝 Alt soru bağlam uzunluğu:", log.split(" - ")[-1].strip())
                        elif "Formatted answers length" in log:
                            st.write("📋 Birleştirilen cevaplar uzunluğu:", log.split(" - ")[-1].strip())
                        elif "Full context being sent to model" in log:
                            st.write("📤 Modele gönderilen tam bağlam:", log.split(" - ")[-1].strip())
                        elif "Full formatted answers being sent to model" in log:
                            st.write("📤 Modele gönderilen tam cevaplar:", log.split(" - ")[-1].strip())
                
                return response
                
            except Exception as e:
                logging.error(f"Soru işleme hatası: {str(e)}")
                return "Üzgünüm, Netpak'ın dokümanlarından yanıt oluştururken bir hata oluştu. Lütfen tekrar deneyin."
        
        return process_question

    except Exception as e:
        logging.error(f"RAG zinciri oluşturma hatası: {str(e)}")
        raise

def format_docs(docs):
    """Format documents with improved structure and error handling"""
    try:
        formatted_docs = []
        for doc in docs:
            try:
                # Metadata'yı daha detaylı ekle
                metadata_str = " | ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                content = doc.page_content.strip()
                
                # DOCX dosyaları için özel formatlama
                if doc.metadata.get("file_type") == "docx":
                    # Paragraf numarasını ekle
                    para_num = doc.metadata.get("chunk_index", 0) + 1
                    formatted_docs.append(
                        f"[DOKÜMAN: {doc.metadata.get('source', 'unknown')} - Paragraf {para_num}]\n"
                        f"{content}\n"
                        f"[Metadata: {metadata_str}]\n"
                    )
                else:
                    formatted_docs.append(f"{content}\n[Metadata: {metadata_str}]")
            except Exception as e:
                logger.error(f"Error formatting document: {str(e)}")
                continue
        
        # Dokümanları sırala ve birleştir
        return "\n\n".join(formatted_docs)
    except Exception as e:
        logger.error(f"Error in format_docs: {str(e)}")
        return "Error formatting documents. Please try again."

def get_docx_files():
    """Get all docx files from the data directory"""
    try:
        docx_files = []
        for file in os.listdir(DATA_DIR):
            if file.endswith('.docx'):
                docx_files.append(file)
        return docx_files
    except Exception as e:
        logger.error(f"Error getting docx files: {str(e)}")
        return []

def streamlit_app():
    """Main Streamlit application with improved UI and performance"""
    try:
        # Configure page
        st.set_page_config(
            page_title="NetPak AI Assistant",
            page_icon="📦",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for professional design
        st.markdown("""
        <style>
        /* Global Styles */
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
            padding-bottom: 100px; /* Add padding for fixed input */
        }
        
        /* Fixed Chat Input */
        .stChatInput {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f8f9fa;
            padding: 1rem;
            z-index: 1000;
            border-top: 1px solid #e5e5e5;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        }
        
        .stChatInput > div {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Chat Container */
        .chat-container {
            margin-bottom: 80px; /* Space for fixed input */
            padding-bottom: 1rem;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            border-bottom: 1px solid #e5e5e5;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px 8px 0 0;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            color: #86868b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #0066cc;
            border-bottom: 2px solid #0066cc;
        }
        
        /* Chat Messages */
        .chat-message {
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .chat-message.user {
            background-color: #f5f5f7;
            margin-left: 2rem;
        }
        
        .chat-message.assistant {
            background-color: #ffffff;
            border: 1px solid #e5e5e5;
            margin-right: 2rem;
        }
        
        /* Buttons */
        .stButton button {
            width: 100%;
            border-radius: 8px;
            height: 2.5rem;
            font-weight: 500;
            background-color: #0066cc;
            color: white;
            border: none;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background-color: #0077ed;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* File Uploader */
        .file-uploader {
            border: 2px dashed #d2d2d7;
            border-radius: 12px;
            padding: 1.5rem;
            background-color: #f5f5f7;
            transition: all 0.2s ease;
        }
        
        .file-uploader:hover {
            border-color: #0066cc;
            background-color: #f0f0f2;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f5f5f7;
        }
        
        /* Success Messages */
        .stSuccess {
            background-color: #e8f5e9;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #81c784;
        }
        
        /* Error Messages */
        .stError {
            background-color: #ffebee;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #e57373;
        }
        
        /* Input Fields */
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #d2d2d7;
            padding: 0.5rem 1rem;
        }
        
        .stTextInput input:focus {
            border-color: #0066cc;
            box-shadow: 0 0 0 2px rgba(0,102,204,0.2);
        }
        </style>
        """, unsafe_allow_html=True)

        # Initialize session state
        if "memory" not in st.session_state:
            st.session_state.memory = EnhancedConversationMemory()

        if "document_manager" not in st.session_state:
            try:
                # Initialize document manager with vector store directory
                st.session_state.document_manager = CorporateDocumentManager(
                    data_dir=DATA_DIR,
                    metadata_dir=METADATA_DIR,
                    vector_store_dir=VECTOR_STORE_DIR,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                # Process documents and initialize vector store
                if st.session_state.document_manager.process_documents():
                    st.success("📚 Knowledge base initialized successfully!")
                else:
                    st.warning("📝 No documents found in the data directory. Please upload some documents.")
            except Exception as e:
                st.error(f"❌ Error initializing knowledge base: {str(e)}")
                logger.error(f"Error initializing knowledge base: {e}")

        # Sidebar with improved organization
        with st.sidebar:
            st.title("📦 NetPak AI Assistant")
            st.markdown("---")
            
            # Database Status Section
            st.subheader("📊 System Status")
            if "document_manager" in st.session_state:
                try:
                    db_stats = st.session_state.document_manager.get_document_stats()
                    st.json(db_stats)
                except Exception as e:
                    st.error(f"❌ Error getting database stats: {str(e)}")
                    logger.error(f"Error getting database stats: {e}")
            
            # Document Processing Section
            st.subheader("📁 Document Processing")
            
            # Display existing docx files
            st.subheader("📚 Document Library")
            docx_files = get_docx_files()
            if docx_files:
                for file in docx_files:
                    st.write(f"📄 {file}")
            else:
                st.info("📝 No documents found in the library.")
            
            uploaded_file = st.file_uploader("📤 Upload Document", type=list(SUPPORTED_EXTENSIONS.keys()))
            if uploaded_file:
                try:
                    # Save the uploaded file
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    logger.error(f"Error processing file {uploaded_file.name}: {e}")
            
            # Update Knowledge Base button
            if st.button("🔄 Update Knowledge Base"):
                with st.spinner("🔄 Updating knowledge base..."):
                    try:
                        if st.session_state.document_manager.process_documents(force_update=True):
                            st.success("✅ Knowledge base updated successfully!")
                        else:
                            st.warning("📝 No documents found to update.")
                    except Exception as e:
                        st.error(f"❌ Error updating knowledge base: {str(e)}")
                        logger.error(f"Error updating knowledge base: {e}")

        # Main content area with tabs
        st.title("📦 NetPak AI Assistant")
        
        # Create tabs
        tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])
        
        with tab1:
            st.markdown("### Ask me anything about your documents!")
            
            # Chat container with custom styling
            chat_container = st.container()
            with chat_container:
                # Display chat history with improved styling
                for message in st.session_state.memory.get_streamlit_messages():
                    if message["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(message["content"])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(message["content"])
            
            # Add some space at the bottom
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Chat input with improved styling - now fixed at bottom
            user_prompt = st.chat_input("💭 Ask me anything...", key="chat_input")
            
            if user_prompt:
                # Add user message to memory and display
                st.session_state.memory.add_message("user", user_prompt)
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_prompt)
                
                # Get AI response
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    with st.spinner("🤔 Thinking..."):
                        try:
                            rag_chain = create_advanced_rag_chain()
                            if rag_chain:
                                response = rag_chain(user_prompt)
                                message_placeholder.markdown(response)
                                st.session_state.memory.add_message("assistant", response)
                            else:
                                st.error("❌ Failed to create RAG chain. Please try updating the knowledge base.")
                        except Exception as e:
                            error_msg = f"I apologize, but I encountered an error: {str(e)}"
                            message_placeholder.markdown(error_msg)
                            st.session_state.memory.add_message("assistant", error_msg)
                            logger.error(f"RAG error: {e}")
        
        with tab2:
            st.markdown("### 📊 Knowledge Base Analytics")
            if "document_manager" in st.session_state:
                try:
                    # Display vector store statistics
                    vector_stats = st.session_state.document_manager.get_vector_store_stats()
                    st.json(vector_stats)
                    
                    # Display conversation memory statistics
                    memory_stats = st.session_state.memory.get_statistics()
                    st.markdown("### 💬 Conversation Memory")
                    st.json(memory_stats)
                except Exception as e:
                    st.error(f"❌ Error displaying statistics: {str(e)}")
                    logger.error(f"Error displaying statistics: {e}")

    except Exception as e:
        logger.error(f"Streamlit app error: {str(e)}")
        st.error("❌ An unexpected error occurred. Please try refreshing the page.")

if __name__ == "__main__":
    try:
        import sys
        if 'streamlit' in sys.modules:
            # Ngrok tünelini başlat
            try:
                # Ngrok auth token'ı ayarla
                ngrok.set_auth_token("2wrgvQA5kXsPUtDAGiRIdkJtUq2_2W4WJeiP9PpRJVncdwoZG")
                
                # Önceki tünelleri temizle
                ngrok.kill()
                
                # Streamlit portu için tünel oluştur
                public_url = ngrok.connect(STREAMLIT_PORT, "http")
                
                # Tünel durumunu kontrol et
                tunnels = ngrok.get_tunnels()
                if tunnels:
                    logger.info(f"Ngrok tunnel established at: {public_url}")
                    print(f"\n🌐 Global URL: {public_url}")
                    print("⚠️  Note: This URL will change each time you restart the application")
                else:
                    raise Exception("No active tunnels found")
                    
            except Exception as e:
                logger.error(f"Error establishing ngrok tunnel: {str(e)}")
                print("\n❌ Failed to establish ngrok tunnel. Running locally only.")
                print("Possible reasons:")
                print("1. Invalid ngrok auth token")
                print("2. Port 8501 is already in use")
                print("3. Network connectivity issues")
                print("4. Ngrok service is down")
                print("\nTo fix:")
                print("1. Check your ngrok auth token")
                print("2. Make sure port 8501 is available")
                print("3. Check your internet connection")
                print("4. Try running 'ngrok http 8501' manually in terminal")
            
            streamlit_app()
        else:
            print("Please run this script using Streamlit: streamlit run demo_v2.1.py")
    except Exception as e:
        logger.error(f"Execution error: {e}")