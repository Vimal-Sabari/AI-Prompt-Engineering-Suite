import os
import sys
import numpy as np
import time
from typing import List, Dict, Tuple

# -------------------- SPACY IMPORT --------------------
try:
    import spacy
    from spacy import displacy
    SPACY_MODEL_NAME = "en_core_web_sm"
    SPACY_SUPPORT = True
except ImportError:
    print("Warning: spaCy not installed.")
    SPACY_SUPPORT = False

# -------------------- TTS IMPORT --------------------
try:
    from gtts import gTTS
    import playsound
    TTS_SUPPORT = True
except ImportError:
    print("Warning: gTTS / playsound not installed.")
    TTS_SUPPORT = False

# -------------------- GEMINI IMPORT --------------------
try:
    import google.generativeai as genai
except ImportError:
    print("FATAL: google-generativeai not installed.")
    genai = None

# -------------------- OPTIONAL NLP IMPORT --------------------
try:
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cosine
    NER_SUPPORT = True
    CLUSTERING_SUPPORT = True
except ImportError:
    NER_SUPPORT = False
    CLUSTERING_SUPPORT = False


# -------------------- CONFIG --------------------

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
DEFAULT_TEMPERATURE = 0.7

LANGUAGES_MAP = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'kn': 'Kannada'
}

# ✅ SAFE API KEY LOADING
API_KEY = os.getenv("GEMINI_API_KEY")


# -------------------- GEMINI INIT --------------------

def initialize_gemini():

    if genai is None:
        return None

    if not API_KEY:
        print("❌ ERROR: GEMINI_API_KEY not found in environment.")
        print("Set it before running the program.")
        return None

    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        model.count_tokens("test")
        print("✅ Gemini initialized successfully.")

        return model

    except Exception as e:
        print("❌ Gemini Init Error:", e)
        return None


gemini_model = initialize_gemini()


# -------------------- SPACY INIT --------------------

SPACY_NLP = None

if SPACY_SUPPORT:
    try:
        SPACY_NLP = spacy.load(SPACY_MODEL_NAME)
        print("✅ spaCy Loaded")
    except:
        print("❌ spaCy model missing")
        SPACY_SUPPORT = False


# -------------------- EMBEDDINGS --------------------

def get_embeddings(texts):

    if not genai:
        return []

    try:
        res = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts
        )

        return res["embedding"]

    except Exception as e:
        print("Embedding Error:", e)
        return []


# -------------------- GEMINI CALL --------------------

def run_gemini(system, user, lang, temp=DEFAULT_TEMPERATURE):

    if not gemini_model:
        return "Gemini not initialized."

    target = LANGUAGES_MAP.get(lang, "English")

    lang_rule = f"You must respond only in {target}."

    prompt = f"{system}\n{lang_rule}\n{user}"

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": temp}
        )

        return response.text

    except Exception as e:
        return f"Error: {e}"


# -------------------- TTS --------------------

def speak(text, lang):

    if not TTS_SUPPORT:
        return

    clean = " ".join(text.split())

    audio = f"tts_{os.getpid()}.mp3"

    try:
        tts = gTTS(clean, lang=lang)
        tts.save(audio)

        playsound.playsound(audio)

    except Exception as e:
        print("TTS Error:", e)

    finally:
        if os.path.exists(audio):
            os.remove(audio)


# -------------------- FEATURES --------------------

def enhance_prompt(prompt, lang):

    system = "You are a prompt engineer. Improve this prompt."
    user = f"Enhance:\n{prompt}"

    return run_gemini(system, user, lang)


def explain_influence(prompt, element, lang):

    system = "You are an analyst."

    user = f"""
Prompt: {prompt}
New Element: {element}
Explain impact.
"""

    return run_gemini(system, user, lang, 0.1)


def generate_content(prompt, lang):

    system = "You are a creative assistant."
    return run_gemini(system, prompt, lang, 0.8)


def score_prompt(prompt, lang):

    system = """
You are a quality rater.
Give score: SCORE X/10
Explain.
"""

    user = f"Rate:\n{prompt}"

    return run_gemini(system, user, lang, 0.0)


def semantic_search(query, corpus, k=3):

    if not CLUSTERING_SUPPORT:
        return []

    texts = [query] + corpus

    emb = get_embeddings(texts)

    if not emb:
        return []

    q = np.array(emb[0])
    docs = np.array(emb[1:])

    scores = []

    for i, d in enumerate(docs):
        sim = 1 - cosine(q, d)
        scores.append((corpus[i], sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:k]


def toxicity_check(prompt):

    system = """
Classify SAFE or HARMFUL.
Format: CLASS | SCORE
"""

    return run_gemini(system, prompt, "en", 0.0)


def ner_visualize(text):

    if not SPACY_SUPPORT:
        print("spaCy disabled")
        return

    doc = SPACY_NLP(text)

    file = f"ner_{int(time.time())}.html"

    html = displacy.render(doc, style="ent", page=True)

    with open(file, "w", encoding="utf-8") as f:
        f.write(html)

    print("Saved:", file)


# -------------------- LANGUAGE --------------------

def get_language():

    print("\nSelect Language:")

    for k, v in LANGUAGES_MAP.items():
        print(k, "-", v)

    while True:

        code = input("Enter code: ").strip().lower()

        if code in LANGUAGES_MAP:
            return code

        print("Invalid.")


# -------------------- MAIN MENU --------------------

def main():

    if not gemini_model:
        return

    print("\n🌐 Multilingual Prompt Toolkit")

    lang = get_language()

    print("Language:", LANGUAGES_MAP[lang])

    while True:

        print("\nMENU")
        print("1. Enhance Prompt")
        print("2. Explain Influence")
        print("3. Score Prompt")
        print("4. Generate Content")
        print("5. Semantic Search")
        print("6. Toxicity Check")
        print("7. NER Visualization")
        print("8. Exit")

        ch = input("Choice: ")

        if ch == "1":

            p = input("Prompt: ")
            out = enhance_prompt(p, lang)

            print(out)

            if TTS_SUPPORT:
                if input("Speak? (y/n): ") == "y":
                    speak(out, lang)


        elif ch == "2":

            p = input("Prompt: ")
            e = input("New Element: ")

            print(explain_influence(p, e, lang))


        elif ch == "3":

            p = input("Prompt: ")
            print(score_prompt(p, lang))


        elif ch == "4":

            p = input("Prompt: ")
            out = generate_content(p, lang)

            print(out)


        elif ch == "5":

            q = input("Query: ")

            n = int(input("Docs count: "))

            corpus = []

            for i in range(n):
                corpus.append(input(f"Text {i+1}: "))

            res = semantic_search(q, corpus)

            for t, s in res:
                print(f"{s:.4f} -> {t}")


        elif ch == "6":

            p = input("Prompt: ")
            print(toxicity_check(p))


        elif ch == "7":

            t = input("English Text: ")
            ner_visualize(t)


        elif ch == "8":

            print("Bye 👋")
            break


        else:
            print("Invalid")


# -------------------- RUN --------------------

if __name__ == "__main__":
    main()
