# AI Study Buddy

AI Study Buddy is a **local-first, offline-capable AI-powered tool** that helps students learn smarter — not harder.

---

## Features

- **Chat with Notes**: Upload any PDF (e.g., class notes) and ask questions about it.
- **Chat Freely**: A smart chatbot to handle general questions and explain concepts.
- **Note Summarizer**: Get clean, bullet-point summaries of your uploaded notes.
- **Quiz Generator**: Auto-generate multiple-choice questions for self-testing.
- **Flashcards**: Turn your notes into Q&A-style flashcards.
- **Study Planner**: Get a customized day-wise study plan based on your subjects.

---

## Models Supported

All models run **locally** using [Ollama](https://ollama.com/) (must be installed):
- `mistral`
- `tinyllama`
- `llama3`

You can choose your model from the sidebar.

---

##  Installation & Setup

### 1. **Clone the repository**
   
        git clone https://github.com/sid-2672/AI-study-buddy.git
        cd AI-study-buddy

### 2. **Create and activate a virtual environment**

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. **Install dependencies**

       pip install -r requirements.txt

### 4. **Install and run Ollama**

   Download from: https://ollama.com

   Once installed, run your desired model:

       ollama run mistral
       # or
       ollama run llama3
        # or
       ollama run tinyllama
### 5. **Run the app**

       cd App
       streamlit run main.py
## Contributing?
  Have an idea to make this app even more awesome?
  Feel free to fork this repo and create a pull request!
## Disclaimer!
 All processing is local — your notes and chats stay private on your machine!
 Make sure you have a good GPU for faster processing. 
 
