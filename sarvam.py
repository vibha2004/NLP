import requests
import time
from typing import Optional
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import pandas as pd
import os

# Sarvam API constants
SARVAM_API_KEY = "sk_zdw93n1y_8xmrHBp5glmO0TcpL8JZY8aF"
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"

REPEAT_RUNS = 10
EXCEL_FILENAME = "sarvam_latency_metrics.xlsx"

LANGUAGE_MAP = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'ml': 'Malayalam', 'kn': 'Kannada', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'or': 'Odia', 'en': 'English'
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

def translate_back(text: str, target_lang: str) -> (str, float):
    max_chunk_size = 4500
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    start = time.time()
    translated_parts = [
        GoogleTranslator(source='en', target=target_lang).translate(chunk)
        for chunk in chunks
    ]
    end = time.time()
    return ' '.join(translated_parts), (end - start) * 1000

def get_sarvam_response(prompt: str, api_key: str) -> (str, float):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sarvam-m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    start = time.time()
    response = requests.post(SARVAM_API_URL, headers=headers, json=payload)
    end = time.time()

    try:
        result = response.json()
    except ValueError:
        return "[Error parsing response]", (end - start) * 1000

    if response.status_code != 200:
        return f"[Error {response.status_code}: {result.get('error', {}).get('message', response.text)}]", (end - start) * 1000

    try:
        return result['choices'][0]['message']['content'].strip(), (end - start) * 1000
    except (KeyError, IndexError):
        return "[Unexpected response format]", (end - start) * 1000

def log_to_excel(data: list, filename: str = EXCEL_FILENAME):
    df = pd.DataFrame(data)

    # Calculate mean values and append as a new row
    mean_row = {
        "Run": "Mean",
        "Input Language": df["Input Language"].iloc[0],
        "Prompt Length": round(df["Prompt Length"].mean(), 2),
        "LLM Response Length": round(df["LLM Response Length"].mean(), 2),
        "To English Latency (ms)": round(df["To English Latency (ms)"].mean(), 2),
        "LLM Inference Latency (ms)": round(df["LLM Inference Latency (ms)"].mean(), 2),
        "To Native Latency (ms)": round(df["To Native Latency (ms)"].mean(), 2)
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(filename, index=False)

if __name__ == "__main__":
    user_input = input("Enter your prompt in any Indian language: ")

    lang_code = detect_language(user_input)
    if lang_code is None:
        print("Could not detect language.")
        exit()

    lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
    print(f"\nDetected Language: {lang_name} ({lang_code})")

    log_data = []

    for run in range(REPEAT_RUNS):
        print(f"\n--- Run {run + 1} ---")

        # Native to English
        translated_english, latency_to_english = translate_to_english(user_input, lang_code)
        print(f"Translated to English: {translated_english} (Latency: {latency_to_english:.2f} ms)")

        # LLM response
        sarvam_output_english, sarvam_latency = get_sarvam_response(translated_english, SARVAM_API_KEY)
        print(f"Sarvam Response: {sarvam_output_english} (Latency: {sarvam_latency:.2f} ms)")

        # English to Native
        translated_back, latency_to_native = translate_back(sarvam_output_english, lang_code)
        print(f"Back Translated: {translated_back} (Latency: {latency_to_native:.2f} ms)")

        # Record data
        log_data.append({
            "Run": run + 1,
            "Input Language": lang_name,
            "Prompt Length": len(user_input),
            "LLM Response Length": len(sarvam_output_english),
            "To English Latency (ms)": round(latency_to_english, 2),
            "LLM Inference Latency (ms)": round(sarvam_latency, 2),
            "To Native Latency (ms)": round(latency_to_native, 2)
        })

    log_to_excel(log_data)
    print(f"\nAll {REPEAT_RUNS} runs completed. Metrics saved to '{EXCEL_FILENAME}'.")

