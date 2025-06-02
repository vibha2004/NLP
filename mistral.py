import time
import requests
import pandas as pd
from typing import Optional, List
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Constants
HF_TOKEN = ""
REPEAT_RUNS = 10
LANGUAGE_MAP = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'ml': 'Malayalam', 'kn': 'Kannada', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'or': 'Odia', 'en': 'English'
}

# Detect language
def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None

# Translate native language to English
def translate_to_english(text: str, source_lang: str) -> str:
    return GoogleTranslator(source=source_lang, target='en').translate(text)

# Translate English back to native language
def translate_back(text: str, target_lang: str) -> str:
    max_chunk_size = 4500
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    translated_parts = [
        GoogleTranslator(source='en', target=target_lang).translate(chunk)
        for chunk in chunks
    ]
    return ' '.join(translated_parts)

# Call LLM (Mistral via HuggingFace API)
def get_llm_response_from_api(prompt: str, hf_token: str) -> str:
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text'].replace(prompt, '').strip()
        elif isinstance(result, list):
            return result[0].get('generated_text', '[No output]')
        else:
            return str(result)
    else:
        return f"API Error {response.status_code}: {response.text}"

# Run the experiment multiple times and record metrics
def run_experiments(user_input: str, repeat: int = 3):
    records = []

    lang_code = detect_language(user_input)
    if lang_code is None:
        print("Could not detect language.")
        return

    lang_name = LANGUAGE_MAP.get(lang_code, lang_code)

    for run in range(repeat):
        print(f"\nRun {run + 1}")

        # Translate to English
        start = time.time()
        translated_english = translate_to_english(user_input, lang_code)
        latency_to_english = (time.time() - start) * 1000
        print(f"Translated to English: {translated_english} [{latency_to_english:.2f} ms]")

        # Get LLM Response
        start = time.time()
        sarvam_output_english = get_llm_response_from_api(translated_english, HF_TOKEN)
        sarvam_latency = (time.time() - start) * 1000
        print(f"LLM Response in English: {sarvam_output_english} [{sarvam_latency:.2f} ms]")

        # Translate back to native language
        start = time.time()
        translated_back = translate_back(sarvam_output_english, lang_code)
        latency_to_native = (time.time() - start) * 1000
        print(f"Translated back to {lang_name}: {translated_back} [{latency_to_native:.2f} ms]")

        # Store metrics
        records.append({
            "Run": run + 1,
            "Input Language": lang_name,
            "Prompt Length": len(user_input),
            "LLM Response Length": len(sarvam_output_english),
            "To English Latency (ms)": round(latency_to_english, 2),
            "LLM Inference Latency (ms)": round(sarvam_latency, 2),
            "To Native Latency (ms)": round(latency_to_native, 2),
            
        })

    return records

# Save results to Excel with a mean row
def save_with_mean_to_excel(results: List[dict], file_name: str):
    df = pd.DataFrame(results)

    mean_row = {
        "Run": "Mean",
        "Input Language": df["Input Language"].iloc[0],
        "Prompt Length": df["Prompt Length"].mean(),
        "LLM Response Length": df["LLM Response Length"].mean(),
        "To English Latency (ms)": round(df["To English Latency (ms)"].mean(), 2),
        "LLM Inference Latency (ms)": round(df["LLM Inference Latency (ms)"].mean(), 2),
        "To Native Latency (ms)": round(df["To Native Latency (ms)"].mean(), 2),
        
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_excel(file_name, index=False)
    print(f"\nMetrics with mean saved to '{file_name}'")

# Main block
if __name__ == "__main__":
    user_input = input("Enter text in any Indian language: ").strip()
    results = run_experiments(user_input, REPEAT_RUNS)

    if results:
        save_with_mean_to_excel(results, "mistral_latency_metrics.xlsx")
