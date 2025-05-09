# AI Assistant

A sophisticated document-based AI assistant built with Streamlit, designed for corporate document management and intelligent conversation.

## Overview

NetPak AI Assistant is an advanced document management and conversational AI system that combines the power of OpenAI's GPT models with document processing capabilities. It provides an intuitive interface for document management, intelligent document search, and context-aware conversations.

## Features

- Document Management
  - Support for multiple document formats (DOCX, PDF, TXT)
  - Automatic document processing and indexing
  - Vector-based document storage for efficient retrieval

- Intelligent Conversation
  - Context-aware responses based on document content
  - Multi-language support (English and Turkish)
  - Advanced conversation memory with vector store integration

- User Interface
  - Clean and professional Streamlit-based interface
  - Real-time document processing status
  - Interactive chat interface
  - Document analytics dashboard

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Ngrok account (for public access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ydemirki/assistant_with_RAG.git
cd assistant_with_RAG
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export NGROK_AUTH_TOKEN="your-ngrok-token"
```

## Project Structure

```
.
├── data/                  # Document storage directory
├── metadata/             # Document metadata storage
├── vector_store/         # Vector database storage
├── demo_v2.1.py         # Main application file
├── document_processor.py # Document processing module
├── document_manager.py   # Document management module
└── requirements.txt      # Project dependencies
```

## Usage

1. Start the application:
```bash
streamlit run demo_v2.1.py
```

2. Access the application:
   - Local: http://localhost:8501
   - Public: URL provided by ngrok (if configured)

3. Upload documents through the sidebar interface

4. Interact with the AI assistant through the chat interface

## Configuration

The application can be configured through the following constants in `demo_v2.1.py`:

- `DATA_DIR`: Document storage directory
- `METADATA_DIR`: Metadata storage directory
- `VECTOR_STORE_DIR`: Vector database directory
- `CHUNK_SIZE`: Document chunk size for processing
- `CHUNK_OVERLAP`: Overlap between document chunks
- `MAX_TOKEN_LIMIT`: Maximum token limit for conversation memory

## Security Considerations

- API keys should be stored securely and never committed to version control
- Document access should be properly authenticated in production environments
- Regular security audits should be performed

## Performance Optimization

- Document chunking parameters can be adjusted based on document size and complexity
- Vector store persistence can be configured for optimal performance
- Conversation memory limits can be adjusted based on usage patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
