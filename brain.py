import chromadb
import datetime
import uuid
from google import genai
import os
from dotenv import load_dotenv
import json
from google.genai import errors
from google.genai._gaos.lib.compat_errors import RateLimitError, InternalServerError

# Load the .env file into the system environment (this contains the API key)
load_dotenv()

# Creates/opens a ChromaDB client that stores the data in the physical disk specified
# Used "PersistentClient" because I wanted the data to persist between executions of the code and the app is for personal local use only, 
# so local storage works fine
db_client = chromadb.PersistentClient("./chroma_storage")

# Retrieves the "developer_brain" collection, creates it if it doesn't exist
# Data is stored semantically (as embeddings), allowing facts to be retrieved by meaning rather than by exact keyword match
collection = db_client.get_or_create_collection(name="developer_brain")

# Initializes a client for the Google GenAI API using credentials from the environment
# Parentheses were left empty because the API key was loaded as a system variable
client = genai.Client()

# Saves the user input into the database as a new memory entry
# The embedding is generated automatically by Chroma's default embedder (all-MiniLM-L6-v2) 
# Input must be in English, as the embedder was trained with English texts, so it might be less accurate in other languages
def save_to_memory(text, category):
    collection.add(
    ids=[str(uuid.uuid4())], # Generates a random unique ID for each entry
    documents=[text], # The user input, stored as the raw text
    metadatas=[{
        "date": datetime.datetime.now().isoformat(), # Tracks when the statement was made by assigning the current date to it
        "category": category # Used to filter/retrieve facts faster later
        }]
    )

# Searches for the memory entry closest in meaning to the user's query, so it can be used as additional context for the AI's response     
def search_memory(query, category):
    query_kwargs = {
        "query_texts": [query], # Chroma converts this text into an embedding to compare it to other stored vectors
        "n_results": min(1, collection.count()) # Retrieve only the closest match; min() avoids an error when the collection is empty
    }

    if category is not None: # If a category is provided, filter results to that category to retrieve the fact faster
        query_kwargs["where"] = {"category": category}
    
    results = collection.query(**query_kwargs) # Unpacks the query_kwargs dictionary and runs the query
    
    try:
        docs = results.get("documents") or [] # "documents" key may be missing or empty; default to an empty list either way to prevent a TypeError
        return docs[0][0] # Return the closest matching entry's text
    except IndexError:
        return "Memory not found!" # Happens when the "documents" key is missing and we try to index the empty list we assigned


# Uses Gemini API to classify the user's question into one of the fixed set of categories
# The category is later used as metadata, allowing a faster and more precise retrieval with filtering
def determine_category(question):
    prompt = f"""
    Analyze the following user question and classify it into exactly ONE of these categories:
    - "interests" (if they are asking about hobbies, sports, preferences, food, etc.)
    - "academic" (if they are asking about university, classes, calculus, studying, etc.)
    - "code_snippets" (if they are asking about python, database, or programming)
    - "none" (if it is a general question like greeting you, math, or general knowledge)

    User question: "{question}"

    Respond with ONLY the category name in lowercase. Do not write any other text.
    """

    # Connects the program to the Gemini API, sends the classification prompt, and requests a response
    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=prompt
        )
    except (errors.APIError, RateLimitError, InternalServerError): # Prevents crash in case the token limit was exceeded or the API is down
        return None

    # Extracts the category text, normalized to lowercase with no extra whitespace
    category = interaction.output_text.strip().lower() # type: ignore

    return None if category == "none" else category

# Generates an answer to the user's question, using relevant context retrieved from ChromaDB when available
def generate_answer(question, previous_interaction):
    defined_category = determine_category(question) # Gets the category of the user's question to filter by metadata if possible
    
    context = search_memory(question, defined_category) # Retrieves the most relevant memory entry, filtering by category

    full_prompt = f"""
        You are a personal assistant.
        Here is a retrieved context/memory from the user: {context}

        Use this memory ONLY if it is relevant to the user's question.
        If the memory is "Memory not found!" or irrelevant, answer normally using your general knowledge.

        User question: {question}
    """

    # Used a dictionary instead of passing the kwargs directly, since previous_interaction_id
    # needs to be conditionally appended only if a history ID exists 
    kwargs = {
        "model": "gemini-3.5-flash",
        "input": full_prompt
    }

    if previous_interaction:
        kwargs["previous_interaction_id"] = previous_interaction # Links the request to the previous one, allowing the API to maintain a continuous conversation

    # Unpacks the "kwargs" dictionary, sends the full_prompt to the Gemini API, and requests a response
    try:
        interaction = client.interactions.create(**kwargs) # type: ignore
    except RateLimitError:
        return "Reached the rate limit. Try again later", previous_interaction
    except (errors.APIError, InternalServerError):
        return "Something went wrong with the AI service. Try again later", previous_interaction

    return interaction.output_text, interaction.id # Returns the AI's response and the interaction id, which is used later to maintain conversational continuity

# Uses Gemini API to determine whether the user input is worth extracting. Return a JSON object for better readability and format consistency
def auto_extract(user_input):
    prompt = f"""
    You are a background memory-extraction utility.
    Analyze the user's message: "{user_input}"

    Determine if the user is sharing a personal fact about themselves that is worth remembering long-term (e.g., their hobbies, academic details, coding preferences, plans, project, etc.).
    Do NOT save greetings or casual talk.

    If they shared a fact, respond with a JSON object containing:
    1. "should_save": true
    2. "extracted_fact": A concise, clear summary of the fact
    3. "category": Exactly one of "interests" (if they are asking about hobbies, sports, preferences, food, etc.),
    "academic" (if they are asking about university, classes, calculus, studying, etc.), 
    or "code_snippets" (if they are asking about python, database, or programming).

    If they did NOT share a personal fact, respond with:
    {{
        "should_save": false
    }}

    Respond with ONLY the raw JSON object. Do not include markdown code block formatting (like ```json).
    """

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=prompt
        )
    except (errors.APIError, RateLimitError, InternalServerError):
        return None, None

    # Tries decoding the json object. Wrapped it around a try/except in case the AI hallucinated and returned something else
    try:
        response = json.loads(interaction.output_text) # type: ignore
    except (json.JSONDecodeError, TypeError):
        return None, None

    # Returns (None, None) when the model determined that the fact is NOT worth saving
    if not response.get("should_save"):
        return None, None
    
    fact = response.get("extracted_fact")
    category = response.get("category")
    # Unlike determine_category, there is no "none" category here since that is already filtered out by "should_save"
    valid_categories = {"interests", "academic", "code_snippets"}
    # Validates that "fact" is non-empty and "category" is one of the expected values, protecting the program from an unexpected response from the model
    if fact and category in valid_categories: 
        return fact, category
    
    return None, None

# Retrieves all the stored entries from ChromaDB, so the sidebar can show the user what has been saved so far.
def memory_browser():
    data = collection.get() # Retrieves every entry stored in the "developer_brain" collection. The parentheses is empty because I want all the entries

    # Builds a list of dicts pairing each entry's id, category, and fact — used later for deletion, filtering, and display
    memory = []
    # zip pairs by position, allowing each entry to have their corresponding id, fact, and category
    for doc_id, meta, doc in zip( 
        data.get("ids") or [],
        data.get("metadatas") or [],
        data.get("documents") or []
    ):
        memory.append({
            "id": doc_id, # The ID is used for deleting the entry later on
            "category": meta["category"], # The category is used for metadata filtering
            "fact": doc # The fact is used to show what the entry is about
        })
    
    return memory

# Deletes an entry from ChromaDB using the ID retrieved from memory_browser
def delete_memory(doc_id):
    collection.delete(ids=[doc_id]) # ids expects a list, even when just deleting a single entry
