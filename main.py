import streamlit as st
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from PyPDF2 import PdfReader
import datetime
import os

# Constants
MAX_PDF_SIZE = 10 * 1024 * 1024
AVAILABLE_MODELS = ["tinyllama", "mistral", "llama3"]

# Initialization
def init_state():
    for key in ["doc_text", "upload_success", "response", "chat_mode", "previous_mode", "start_fresh"]:
        if key not in st.session_state:
            st.session_state[key] = ""

# Load AI model dynamically
def load_ai(model_name):
    llm = Ollama(model=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", k=3)

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"],
        template="""
You are a helpful AI tutor. Be concise and clear.

Previous chat:
{chat_history}

Student: {user_input}
Tutor:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))
    return chain, memory, wiki_tool

# Process PDF
def process_pdf(file):
    if file.size > MAX_PDF_SIZE:
        raise ValueError("File too large. 10MB max.")
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    if not text.strip():
        raise ValueError("No readable text found.")
    return text

# Save chat history
def save_history(memory):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("data", exist_ok=True)
    path = f"data/study_chat_{ts}.txt"
    with open(path, "w") as f:
        for msg in memory.chat_memory.messages:
            role = "You" if msg.role == "human" else "Tutor"
            f.write(f"{role}: {msg.get_content()}\n")
    return path

# Summarize notes
def summarize_text(llm, text):
    prompt = f"Summarize the following notes in bullet points:\n{text}"
    return llm.invoke(prompt)

# Generate quiz questions
def generate_quiz(llm, text):
    prompt = f"From the following notes, generate 5 multiple-choice quiz questions:\n{text}"
    return llm.invoke(prompt)

# Flashcard generator
def generate_flashcards(llm, text):
    prompt = f"Create Q&A flashcards from these notes:\n{text}"
    return llm.invoke(prompt)

# Study planner
def plan_schedule(llm, subjects, days):
    prompt = f"Create a day-wise study plan for {subjects} over {days} days."
    return llm.invoke(prompt)

# Simple general-purpose chatbot
def run_general_chat(llm, memory, user_input):
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"],
        template="""
You are a helpful and friendly AI assistant.

Previous chat:
{chat_history}

User: {user_input}
AI:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return chain.run(user_input)

# Main App
def main():
    init_state()
    st.set_page_config(page_title="AI Study Buddy", layout="wide")
    st.title("📘 AI Study Buddy")
    st.caption("Your AI-powered companion for notes, quizzes, summaries, and study planning. Offline-ready.")

    # Sidebar: Model & Upload
    with st.sidebar:
        st.markdown("### 🤖 Choose Your AI Model")
        selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=1)

        st.markdown("### 📤 Upload PDF Notes")
        uploaded_file = st.file_uploader("Choose a file", type="pdf")
        if uploaded_file:
            try:
                st.session_state.doc_text = process_pdf(uploaded_file)
                st.session_state.upload_success = True
                st.success("Notes uploaded successfully.")
            except Exception as e:
                st.error(str(e))

        st.markdown("### 🧹 Chat Settings")
        st.session_state.start_fresh = st.checkbox("Start fresh when switching modes", value=False)

    # Load AI and Tools
    chain, memory, wiki_tool = load_ai(selected_model)
    llm = Ollama(model=selected_model)

    st.markdown("---")
    st.subheader("💬 Ask a Question or Choose a Tool")

    mode = st.radio("Choose Mode", ["Chat with Notes", "Chat Freely", "Summarize", "Generate Quiz", "Flashcards", "Study Planner"])

    # Detect mode change
    if st.session_state.previous_mode != mode:
        if st.session_state.start_fresh:
            memory.clear()
            st.session_state.response = ""
    st.session_state.previous_mode = mode

    if mode == "Chat with Notes" and st.session_state.upload_success:
        with st.form(key="chat_notes_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question based on the notes:", key="notes_input")
            submit = st.form_submit_button("Ask")
            if submit and user_input.strip():
                with st.spinner("Thinking..."):
                    context = f"Notes:\n{st.session_state.doc_text}\n\nQuestion: {user_input}"
                    response = chain.run(context)
                    if "i don't know" in response.lower() or len(response.strip()) < 10:
                        wiki = wiki_tool.run(user_input)
                        if wiki:
                            response = f"🔍 From Wikipedia:\n{wiki}"
                    memory.chat_memory.add_user_message(user_input)
                    memory.chat_memory.add_ai_message(response)
                    st.session_state.response = response
                    st.rerun()

        st.markdown("### 🧠 Tutor Replies")
        for msg in memory.chat_memory.messages:
            speaker = "👨‍🎓 You" if msg.role == "human" else "🎓 Tutor"
            st.markdown(f"**{speaker}:** {msg.get_content()}")

    elif mode == "Chat Freely":
        with st.form(key="chat_free_form", clear_on_submit=True):
            user_input = st.text_input("Ask anything:", key="free_input")
            submit = st.form_submit_button("Send")
            if submit and user_input.strip():
                with st.spinner("Thinking..."):
                    response = run_general_chat(llm, memory, user_input)
                    memory.chat_memory.add_user_message(user_input)
                    memory.chat_memory.add_ai_message(response)
                    st.markdown(f"**🎓 AI:** {response}")

    elif mode == "Summarize" and st.session_state.upload_success:
        if st.button("Summarize Notes"):
            with st.spinner("Summarizing notes..."):
                summary = summarize_text(llm, st.session_state.doc_text)
                st.markdown("### 📌 Summary")
                st.write(summary)

    elif mode == "Generate Quiz" and st.session_state.upload_success:
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz questions..."):
                quiz = generate_quiz(llm, st.session_state.doc_text)
                st.markdown("### 📝 Quiz Questions")
                st.write(quiz)

    elif mode == "Flashcards" and st.session_state.upload_success:
        if st.button("Create Flashcards"):
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(llm, st.session_state.doc_text)
                st.markdown("### 📇 Flashcards")
                st.write(cards)

    elif mode == "Study Planner" and st.session_state.upload_success:
        subjects = st.text_input("Subjects (comma-separated)", "Physics, Chemistry")
        days = st.number_input("Study duration in days", min_value=1, max_value=60, value=7)
        if st.button("Plan My Study"):
            with st.spinner("Planning your schedule..."):
                plan = plan_schedule(llm, subjects, days)
                st.markdown("### 🗓️ Study Plan")
                st.write(plan)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Session"):
            saved = save_history(memory)
            st.success(f"Saved to {saved}")

    with col2:
        if st.button("🧹 Clear Chat"):
            memory.clear()
            st.session_state.response = ""
            st.rerun()

if __name__ == "__main__":
    main()
