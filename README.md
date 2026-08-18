# Personal AI Second Brain

A personal AI assistant with long-term memory. It automatically extracts and remembers facts you share in conversations, using semantic search to recall relevant context in future chats, providing more accurate and personalized responses. Fact extraction, classification, and response generation are all powered by the Gemini API.

## Features

- **Long-term memory** - automatically extracts and stores personal facts shared during conversations.
- **Semantic search** - retrieves relevant context by meaning, not just keyword matching.
- **Contextual responses** - enhances the prompt with relevant stored facts before sending it to Gemini, producing more accurate and personalized answers. 
- **Conversational continuity** - maintains context across messages within the same chat session.
- **Memory browser** - view and delete the facts that have been automatically extracted.

## Tech Stack

- **Python** - main programming language
- **ChromaDB** - vector database that enables semantic search
- **Streamlit** - web interface
- **Gemini API** - fact extraction, classification, and response generation

## Prerequisites

- Python 3.14 installed
- A Gemini API key ([get one here](https://ai.google.dev/gemini-api/docs))

## Installation

1. Clone the repository

2. Set up your own virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
```

3. Install the dependencies
```bash
   pip install -r requirements.txt
```
4. Get your own Gemini API key and set it up in a new .env file. In the new .env file, add:
```
   GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
streamlit run interface.py
```

This runs the web interface in your browser, where you will be able to interact with the assistant.

## Notes/Limitations

- The embedder works best in English. If used with any other language, there might be mismatches at the moment of retrieval
- The app is intended for personal, local use only. Compatibility across the network hasn't been implemented yet.
