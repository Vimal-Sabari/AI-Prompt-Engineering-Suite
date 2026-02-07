🌐 MULTILINGUAL PROMPT ENGINEERING TOOLKIT

An interactive Python application designed to help users design, analyze, optimize, and evaluate prompts for Large Language Models (LLMs) using Google Gemini. It supports multilingual generation, NLP-based analysis, clustering, semantic search, and speech output through a menu-driven interface.

🚀 Built for AI enthusiasts, students, and researchers interested in Prompt Engineering.

✨ FEATURES

✅ Multilingual Prompt Enhancement
✅ Prompt Influence Analysis
✅ Prompt Quality Scoring
✅ Prompt Categorization (Clustering)
✅ Multilingual Content Generation
✅ Semantic Search using Embeddings
✅ Toxicity and Safety Check
✅ Named Entity Recognition (NER) Visualization
✅ Prompt A/B Testing with AI Judge
✅ Text-to-Speech Output

🌍 Supported Languages
English | Hindi | Tamil | Telugu | Malayalam | Kannada

🧠 TECHNOLOGIES USED

🐍 Python 3.x
🤖 Google Gemini API
📘 spaCy
🔍 Transformers
📊 scikit-learn
📐 SciPy
🔊 gTTS
🔢 NumPy

📁 PROJECT STRUCTURE

Project Folder
├── main.py
├── README.txt
└── requirements.txt

⚙️ INSTALLATION

🔹 Step 1: Clone Repository

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name

🔹 Step 2: Install Dependencies

pip install -r requirements.txt

Or manually:

pip install google-generativeai spacy transformers scikit-learn scipy gtts playsound numpy

🔹 Step 3: Download spaCy Model (Optional)

python -m spacy download en_core_web_sm

🔑 API KEY SETUP

This project uses Google Gemini API.

Open main.py and replace:

API_KEY_DIRECT = "YOUR_API_KEY_HERE"

with your actual API key.

⚠️ IMPORTANT: Never expose your real API key in public repositories.

▶️ HOW TO RUN

Run the application using:

python main.py

You will see an interactive menu on startup.

📋 MENU OPTIONS

1️⃣ Prompt Enhancement
2️⃣ Explain Influence & Modified Content
3️⃣ Score Prompt Quality
4️⃣ Categorize Prompts
5️⃣ Generate Content
6️⃣ Semantic Search
7️⃣ Toxicity Check
8️⃣ NER Visualization
9️⃣ Prompt A/B Tester
🔟 Exit

🧩 FUNCTIONAL OVERVIEW

📝 Prompt Enhancement
Transforms simple prompts into detailed prompts.

🔍 Influence Analysis
Explains how added constraints affect output.

⭐ Prompt Scoring
Evaluates prompt quality using AI.

📊 Prompt Categorization
Groups similar prompts using clustering.

🔎 Semantic Search
Finds related prompts using similarity.

🏷️ NER Visualization
Detects named entities and creates HTML output.

⚔️ Prompt A/B Testing
Compares two prompts and selects the best.

🔊 Text-to-Speech
Reads outputs aloud.

📌 SAMPLE USAGE

Select Language → Choose Feature → Enter Prompt → View Output

Example:

Enter Prompt: Write a story about AI
Output: Multilingual generated response

🛠️ OPTIONAL DEPENDENCIES

Some features need extra libraries.

📘 NER → spaCy
📊 Clustering → scikit-learn
🔍 Search → SciPy
🔊 Speech → gTTS

If missing, fallback methods are used.

⚠️ LIMITATIONS

❗ IndicNER model disabled
❗ Internet required
❗ API quota limits
❗ API key security needed

🚀 FUTURE ENHANCEMENTS

🌐 Web Interface
📱 Mobile App
👤 User Authentication
☁️ Cloud Deployment
🌍 More Languages
🔐 Secure Key Storage

👨‍💻 AUTHOR

Vimal Sabari
Computer Science Student
AI & NLP Enthusiast

📜 LICENSE

MIT License
Free to use, modify, and distribute.

⭐ SUPPORT

If you like this project, please give it a ⭐ on GitHub.

For suggestions or issues, open an issue.

Made with ❤️ for AI and Prompt Engineering