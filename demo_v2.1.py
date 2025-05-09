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
import pandas as pd
from pyngrok import ngrok

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
METADATA_DIR = "./metadata"
VECTOR_STORE_DIR = "./vector_store"
MAX_RETRIES = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_TOKEN_LIMIT = 3000
STREAMLIT_PORT = 8501  # Streamlit'in varsayılan portu

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

class EnhancedConversationMemory:
    """Enhanced conversation memory with vector store integration"""
    
    def __init__(self, max_token_limit: int = MAX_TOKEN_LIMIT):
        self.messages: List[Dict[str, Any]] = []
        self.max_token_limit = max_token_limit
        self.token_count = 0
        self.vector_store = Chroma(
            persist_directory=os.path.join(VECTOR_STORE_DIR, "conversation_memory"),
            embedding_function=OpenAIEmbeddings()
        )
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize memory with improved system message"""
        system_message = {
            "role": "system",
            "content": """You are a professional AI assistant for a packaging company. 
            You provide accurate, helpful, and business-focused responses.
            Always maintain context from previous messages and use it to provide more relevant answers.
            If you're unsure about something, ask for clarification rather than making assumptions.""",
            "tokens": len("You are a professional AI assistant for a packaging company.") / 4
        }
        self.add_message(system_message["role"], system_message["content"])
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message with vector store integration"""
        try:
            estimated_tokens = len(content) / 4
            message = {"role": role, "content": content, "tokens": estimated_tokens}
            
            # Vektör veritabanına kaydet
            doc = Document(
                page_content=content,
                metadata={
                    "role": role,
                    "timestamp": datetime.now().isoformat(),
                    "tokens": estimated_tokens
                }
            )
            self.vector_store.add_documents([doc])
            
            if self.token_count + estimated_tokens > self.max_token_limit:
                self._prune_old_messages(estimated_tokens)
            
            self.messages.append(message)
            self.token_count += estimated_tokens
            logger.debug(f"Added message. Current token count: {self.token_count}")
        except Exception as e:
            logger.error(f"Error adding message to memory: {str(e)}")
    
    def _prune_old_messages(self, required_tokens: float) -> None:
        """Prune old messages while maintaining essential context"""
        while self.token_count + required_tokens > self.max_token_limit and len(self.messages) > 2:
            if len(self.messages) > 2:
                removed = self.messages.pop(1)
                self.token_count -= removed["tokens"]
                logger.debug(f"Pruned message. New token count: {self.token_count}")
    
    def get_relevant_messages(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get most relevant messages based on similarity search"""
        try:
            # Benzer mesajları ara
            docs = self.vector_store.similarity_search(query, k=k)
            relevant_messages = []
            
            for doc in docs:
                message = {
                    "role": doc.metadata["role"],
                    "content": doc.page_content,
                    "tokens": doc.metadata["tokens"]
                }
                relevant_messages.append(message)
            
            return relevant_messages
        except Exception as e:
            logger.error(f"Error getting relevant messages: {str(e)}")
            return []
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get chat history with improved context management"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages[1:]]
    
    def get_recent_messages(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get most recent messages with context"""
        return self.messages[-n:] if len(self.messages) >= n else self.messages
    
    def get_streamlit_messages(self) -> List[Dict[str, str]]:
        """Get messages in Streamlit format"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]

def create_advanced_rag_chain(document_manager: CorporateDocumentManager) -> Any:
    """Create an advanced RAG chain with improved reasoning and performance"""
    try:
        def detect_language(text: str) -> str:
            """Detect language of the input text"""
            # Türkçe karakterleri kontrol et
            tr_chars = set('çğıöşüÇĞİÖŞÜ')
            en_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            tr_count = sum(1 for c in text if c in tr_chars)
            en_count = sum(1 for c in text if c in en_chars)
            
            return 'tr' if tr_count > en_count else 'en'
        
        def get_system_prompt(lang: str) -> str:
            """Get system prompt based on detected language"""
            if lang == 'tr':
                return """Profesyonel bir yapay zeka asistanısınız. 
                Göreviniz, verilen bağlamı kullanarak doğru ve yardımcı yanıtlar vermektir.
                Şu kurallara uyun:
                1. Sadece bağlamda verilen bilgileri kullanın
                2. Bağlam yeterli bilgi içermiyorsa, bunu açıkça belirtin
                3. Doküman analizinde, önemli bilgilere odaklanın
                4. Profesyonel ve net bir iletişim tarzı kullanın
                5. Emin olmadığınız bir konuda, varsayımda bulunmak yerine açıklama isteyin
                6. Mümkün olduğunda bağlamdan spesifik örnekler verin
                7. Her zaman en alakalı bilgileri önceliklendirin
                8. Bağlamdaki bilgileri mantıksal bir sırayla sunun"""
            else:
                return """You are a professional AI assistant. 
                Your task is to provide accurate and helpful responses based on the provided context.
                Follow these guidelines:
                1. Use only the information provided in the context
                2. If the context doesn't contain enough information, say so clearly
                3. For document analysis, focus on extracting key information
                4. Maintain a professional and clear communication style
                5. If you're unsure about something, ask for clarification
                6. Provide specific examples from the context when possible
                7. Always prioritize the most relevant information
                8. Present information in a logical sequence"""
        
        def get_user_prompt(context: str, question: str, lang: str) -> str:
            """Get user prompt based on detected language"""
            if lang == 'tr':
                return f"""Bağlam: {context}

Soru: {question}

Lütfen yukarıdaki bağlama dayanarak detaylı ve doğru bir yanıt verin.
Önemli noktalar:
1. Sadece soruyla doğrudan ilgili bilgileri kullanın
2. En alakalı bilgileri önceliklendirin
3. Bilgileri mantıksal bir sırayla sunun
4. Bağlam yeterli değilse, hangi ek bilgilerin gerekli olduğunu belirtin
5. Doküman analizinde, en önemli noktalara odaklanın"""
            else:
                return f"""Context: {context}

Question: {question}

Please provide a detailed and accurate response based on the context above.
Key points:
1. Use only information directly relevant to the question
2. Prioritize the most relevant information
3. Present information in a logical sequence
4. If context is insufficient, specify what additional information is needed
5. In document analysis, focus on the most important points"""
        
        def process_with_advanced_reasoning(inputs: Dict[str, Any]) -> str:
            """Process user input with advanced reasoning and improved context handling"""
            try:
                question = inputs["question"]
                chat_history = inputs.get("chat_history", [])

                # Detect language
                detected_lang = detect_language(question)

                # Retrieve context with improved relevance
                retrieved_docs = document_manager.search_documents(question, k=8)
                
                # Get relevant conversation history
                relevant_messages = st.session_state.memory.get_relevant_messages(question, k=5)
                
                if not retrieved_docs and not relevant_messages:
                    if detected_lang == 'tr':
                        return "Dokümanlarda veya konuşma geçmişinde sorunuzu yanıtlamak için yeterli bilgi bulamadım. Lütfen sorunuzu yeniden ifade edin veya daha ilgili dokümanlar yükleyin."
                    else:
                        return "I couldn't find any relevant information in the documents or conversation history to answer your question. Please try rephrasing your question or upload more relevant documents."
                
                # Format context
                context = format_docs(retrieved_docs)
                
                # Add conversation history
                if relevant_messages:
                    conversation_context = "\n\nRelevant conversation history:\n"
                    for msg in relevant_messages:
                        conversation_context += f"{msg['role']}: {msg['content']}\n"
                    context += conversation_context

                # Prepare messages with improved prompting
                messages = [
                    {"role": "system", "content": get_system_prompt(detected_lang)},
                    *chat_history,
                    {"role": "user", "content": get_user_prompt(context, question, detected_lang)}
                ]

                try:
                    # Get response from OpenAI with improved parameters
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.15,  # Daha tutarlı yanıtlar için düşük sıcaklık
                        max_tokens=1500,  # Daha detaylı yanıtlar için token sayısını artır
                        presence_penalty=0.6,
                        frequency_penalty=0.3,
                        top_p=0.95,
                        stop=None
                    )
                    
                    if not response.choices:
                        if detected_lang == 'tr':
                            return "Üzgünüm, bir yanıt oluşturamadım. Lütfen tekrar deneyin."
                        else:
                            return "I apologize, but I couldn't generate a response. Please try again."
                    
                    return response.choices[0].message.content

                except Exception as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    if detected_lang == 'tr':
                        return "Üzgünüm, isteğinizi işlerken bir hata oluştu. Lütfen biraz sonra tekrar deneyin."
                    else:
                        return "I apologize, but I encountered an error while processing your request. Please try again in a moment."

            except Exception as e:
                logger.error(f"Error in RAG processing: {str(e)}")
                if detected_lang == 'tr':
                    return "Üzgünüm, isteğinizi işlerken bir hata oluştu. Lütfen tekrar deneyin veya sorunuzu yeniden ifade edin."
                else:
                    return "I apologize, but I encountered an error processing your request. Please try again or rephrase your question."

        return process_with_advanced_reasoning

    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        return None

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
                            rag_chain = create_advanced_rag_chain(st.session_state.document_manager)
                            if rag_chain:
                                response = rag_chain({
                                    "question": user_prompt,
                                    "chat_history": st.session_state.memory.get_chat_history()
                                })
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
                    memory_stats = {
                        "Total Messages": len(st.session_state.memory.messages),
                        "Current Token Count": st.session_state.memory.token_count,
                        "Memory Location": os.path.join(VECTOR_STORE_DIR, "conversation_memory")
                    }
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