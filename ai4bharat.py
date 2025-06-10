import requests
import time
from typing import Optional
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import pandas as pd
import os

HF_API_TOKEN = ""
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"

REPEAT_RUNS = 3  
EXCEL_FILENAME = "ai4bharat_mcq_only.xlsx"


MILU_LANGUAGES = ['hi', 'ta', 'te', 'bn', 'kn']

LANGUAGE_MAP = {
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'kn': 'Kannada'
}

PROMPT_TYPES = {
    "MCQ": {
        "hi": (
            "प्रश्न: भारत की राजधानी क्या है?\n"
            "a) मुंबई\n"
            "b) दिल्ली\n"
            "c) कोलकाता\n"
            "d) चेन्नई\n"
            "उत्तर: b"
        ),
        "ta": (
            "கேள்வி: இந்தியாவின் தலைநகரம் எது?\n"
            "a) சென்னை\n"
            "b) மும்பை\n"
            "c) டெல்லி\n"
            "d) கொல்கத்தா\n"
            "பதில்: c"
        ),
        "te": (
            "ప్రశ్న: భారత రాజధాని ఏది?\n"
            "a) ముంబై\n"
            "b) ఢిల్లీ\n"
            "c) చెన్నై\n"
            "d) కోల్‌కతా\n"
            "సరైన జవాబు: b"
        ),
        "bn": (
            "প্রশ্ন: ভারতের রাজধানী কোনটি?\n"
            "a) মুম্বাই\n"
            "b) দিল্লি\n"
            "c) কলকাতা\n"
            "d) চেন্নাই\n"
            "সঠিক উত্তর: b"
        ),
        "kn": (
            "ಪ್ರಶ್ನೆ: ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?\n"
            "a) ಮುಂಬೈ\n"
            "b) ದೆಹಲಿ\n"
            "c) ಚೆನ್ನೈ\n"
            "d) ಕೋಲ್ಕತಾ\n"
            "ಸರಿಯಾದ ಉತ್ತರ: b"
        ),
    }
}

def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None

def translate_to_english(text: str, source_lang: str) -> (str, float):
    start = time.time()
    translated = GoogleTranslator(source=source_lang, target='en').translate(text)
    end = time.time()
    return translated, (end - start) * 1000

def get_hf_response(prompt: str, api_token: str) -> (str, float):
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7, "max_new_tokens": 256},
        "options": {"wait_for_model": True}
    }

    start = time.time()
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    end = time.time()

    try:
        result = response.json()
    except ValueError:
        return "[Error parsing response]", (end - start) * 1000

    if response.status_code != 200:
        return f"[Error {response.status_code}: {result.get('error', response.text)}]", (end - start) * 1000

    try:
        generated_text = result[0]['generated_text']
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text, (end - start) * 1000
    except (KeyError, IndexError, TypeError):
        return "[Unexpected response format]", (end - start) * 1000

def log_to_excel(data: list, filename: str = EXCEL_FILENAME):
    df = pd.DataFrame(data)

    mean_row = {
        "Run": "Mean",
        "Prompt": df["Prompt"].iloc[0],
        "Question Type": df["Question Type"].iloc[0],
        "Input Language": df["Input Language"].iloc[0],
        "Prompt Length": round(df["Prompt Length"].mean(), 2),
        "LLM Response Length": round(df["LLM Response Length"].mean(), 2),
        "To English Latency (ms)": round(df["To English Latency (ms)"].mean(), 2),
        "LLM Inference Latency (ms)": round(df["LLM Inference Latency (ms)"].mean(), 2)
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(filename, index=False)

if __name__ == "__main__":
    all_results = []

    for q_type, lang_prompts in PROMPT_TYPES.items():
        for lang_code, native_prompt in lang_prompts.items():
            # Only run MCQ prompts if language is in MILU_LANGUAGES
            if q_type == "MCQ" and lang_code not in MILU_LANGUAGES:
                continue

            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
            print(f"\n=== {q_type} | Language: {lang_name} ({lang_code}) ===")
            run_results = []

            for run in range(REPEAT_RUNS):
                print(f"\n--- Run {run + 1} ---")

                # Translate to English
                translated_english, latency_to_english = translate_to_english(native_prompt, lang_code)
                print(f"To English: {translated_english} ({latency_to_english:.2f} ms)")

                # LLM inference
                hf_output, hf_latency = get_hf_response(translated_english, HF_API_TOKEN)
                print(f"Hugging Face output: {hf_output} ({hf_latency:.2f} ms)")

                run_results.append({
                    "Run": run + 1,
                    "Prompt": native_prompt,
                    "Question Type": q_type,
                    "Input Language": lang_name,
                    "Prompt Length": len(native_prompt),
                    "LLM Response Length": len(hf_output),
                    "To English Latency (ms)": round(latency_to_english, 2),
                    "LLM Inference Latency (ms)": round(hf_latency, 2)
                })

            # Log results
            log_to_excel(run_results)
            all_results.extend(run_results)

    print(f"\nAll MCQ prompts for MILU languages completed. Results saved in '{EXCEL_FILENAME}'.")
