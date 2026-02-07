# 🌐 Multilingual Prompt Engineering Toolkit

A powerful, menu-driven Python application that helps users design, analyze, evaluate, and optimize prompts for Large Language Models (LLMs) using Google Gemini.  
It supports multilingual generation, NLP-based clustering, semantic search, prompt scoring, and speech output.

---

## 🚀 Features

✅ Multilingual Prompt Enhancement  
✅ Prompt Influence Analysis  
✅ Prompt Quality Scoring  
✅ Prompt Categorization (Clustering)  
✅ Multilingual Content Generation  
✅ Semantic Search  
✅ Toxicity & Safety Check  
✅ Named Entity Recognition (NER) Visualization  
✅ Prompt A/B Testing with AI Judge  
✅ Text-to-Speech Output (Optional)

Supported Languages:
- English
- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada

---

## 🧠 Technologies Used

- Python 3.x  
- Google Gemini API  
- spaCy (NER)  
- Transformers (Optional)  
- scikit-learn (Clustering)  
- gTTS (Text-to-Speech)  
- NumPy  
- SciPy  

---

## 📂 Project Structure

project-folder/
│
├── main.py # Main application file
├── README.md # Documentation
└── requirements.txt # Dependencies (recommended)


---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r requirements.txt
Or install manually:

pip install google-generativeai spacy transformers scikit-learn scipy gtts playsound numpy
3️⃣ Download spaCy Model (Optional)
python -m spacy download en_core_web_sm
🔑 API Key Setup (Important)
This project uses the Google Gemini API.

In main.py, replace:

API_KEY_DIRECT = "YOUR_API_KEY_HERE"
with your own API key:

API_KEY_DIRECT = "your_api_key_here"
⚠️ Do NOT expose your real API key in public repositories.
Use environment variables for production.

▶️ How to Run
Run the application using:

python main.py
You will see an interactive menu:

1. Prompt Enhancement
2. Explain Influence
3. Score Prompt Quality
4. Categorize Prompts
5. Generate Content
6. Semantic Search
7. Toxicity Check
8. NER Visualization
9. Prompt A/B Tester
10. Exit
Select an option and follow the instructions.

🧩 Functional Modules
🔹 Prompt Enhancement
Improves simple prompts into detailed, structured prompts.

🔹 Prompt Influence Analysis
Explains how new elements affect output.

🔹 Prompt Quality Scoring
Rates prompt effectiveness using AI-based evaluation.

🔹 Prompt Categorization
Groups similar prompts using embeddings and clustering.

🔹 Semantic Search
Finds similar prompts using cosine similarity.

🔹 NER Visualization
Detects named entities and generates HTML visualization.

🔹 Prompt A/B Testing
Compares two prompts and selects the best using AI judgment.

🔹 Text-to-Speech
Reads outputs aloud using gTTS.

📌 Example Usage
Select Language → Choose Feature → Enter Prompt → Get Output
Example:

Enter Prompt: Write a story about a robot
Output: Enhanced multilingual response
🛠️ Optional Dependencies
Some features require additional libraries:

Feature	Library
NER	spaCy
Clustering	scikit-learn
Semantic Search	SciPy
Speech	gTTS
If missing, the program falls back safely.

⚠️ Known Limitations
IndicNER model is disabled for stability.

Requires stable internet for Gemini API.

API usage may incur costs.

Public repos should hide API keys.

📈 Future Improvements
Web Interface (Flask/React)

User Authentication

Cloud Deployment

More Language Support

API Key Encryption

Mobile App Integration

👨‍💻 Author
Vimal Sabari

Computer Science Student
AI & NLP Enthusiast
Prompt Engineering Researcher

📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute.

⭐ Support
If you like this project, please ⭐ star the repository on GitHub!

For issues or suggestions, feel free to open an issue.