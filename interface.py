import streamlit as st
import brain

# Sets the webpage title and uses a wide layout to fill the whole screen rather than just the center
# It is just a purely stylistic decision, doesn't influence the functionality of the webpage
st.set_page_config(page_title="My Second Brain", layout="wide")
st.title("🧠 Personal AI Second Brain") 

# Initializes "just_deleted" in Streamlit's session state so it survives the constant reruns of the script
# Starts as False, and only becomes True right after the user deletes a memory from the sidebar to trigger a confirmation message
if "just_deleted" not in st.session_state:
    st.session_state["just_deleted"] = False

# Creates a sidebar that contains all the facts stored in ChromaDB
st.sidebar.title("💾 Stored Memory")
# Iterates over the list returned by memory_browser to display each stored entry
for memory in brain.memory_browser():
    # Divides the sidebar in two columns. Since the first one contains the fact, it takes most of the space.
    # Alignment must be "centered" so that the deletion button matches the text
    # Creation of the columns must be inside the for loop for the columns to align
    col1, col2 = st.sidebar.columns([0.7, 0.3], vertical_alignment="center")
    with col1:
        st.write(memory["fact"]) # Displays the fact's text in the sidebar
    # Set a button that triggers the deletion of a memory entry by clicking on it
    with col2:
        # Used the memory's ID as the key to make the each button unique, preventing the webpage from crashing. 
        # The key doesn't need to be the memory's ID specifically, but it was the simplest option, since it is a different value after each iteration
        if st.button("X", type="primary", key=memory["id"]):
            brain.delete_memory(memory["id"])
            st.session_state["just_deleted"] = True # Flags that a deletion just happened, so the toast message shows after the rerun
            st.rerun() # Forces Streamlit to rerun the script immediately, which removes the deleted entry from the sidebar

# After the rerun, "just_deleted" is True, so this block runs once to confirm the deletion, then resets the flag  
if st.session_state["just_deleted"]:
    st.toast("Memory deleted!", icon="🗑️")
    st.session_state["just_deleted"] = False # Resets the flag so the toast doesn't reappear on the next rerun

# Initializes "messages" in session state so previous conversation turns survive each rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-renders the full conversation history on every rerun, since Streamlit doesn't persist UI elements between reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # "role" can either be "assistant" or "user"
        st.markdown(message["content"]) # The actual text of the message

# Lets "previous_ID" to persist between reruns, so that the assistant can maintain a continuous, context-aware conversation
if "previous_id" not in st.session_state:
    st.session_state.previous_id = None

# The walrus operator (:=) assigns the input to "prompt" AND checks it's not empty in the same line
# st.chat_input return None if the user types nothing, so this block only runs when the user types something
if prompt := st.chat_input("What is up?"):
    # Displays the user request in the UI
    with st.chat_message("user"):
        st.markdown(prompt)
    # Appends the "role" and "content" of the requests so that it gets saved before the rerun and can be displayed in the UI
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Uses auto_extract to check whether the prompt contains a fact worth saving long-term
    fact, category = brain.auto_extract(prompt)

    # Only saves if both a fact and category were returned (auto_extract returns (None, None) when there's nothing worth saving)
    if fact and category:
        brain.save_to_memory(fact, category)
        st.markdown(f"Saved: {fact}") # Lets the user know a fact was extracted and stored

    # Calls "generate_answer" to get a response from Gemini, and updates "previous_id" to maintain conversational continuity
    # Unpacks directly into st.session_state.previous_id (instead of using a temporary variable)
    response, st.session_state.previous_id = brain.generate_answer(prompt, st.session_state.previous_id)
    with st.chat_message("assistant"):
        st.markdown(response)

    # Same as in the previous append to the "messages" session state
    st.session_state.messages.append({"role": "assistant", "content": response})