import requests
import time
from typing import Optional
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import pandas as pd
import os

SARVAM_API_KEY = "sk_zdw93n1y_8xmrHBp5glmO0TcpL8JZY8aF"
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"

REPEAT_RUNS = 3  
EXCEL_FILENAME = "sarvam_latency_metrics1.xlsx"

LANGUAGE_MAP = {
    'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu'
}

PROMPT_TYPES = {
    "Q&A": {
        "hi": "ताज महल कहाँ स्थित है?",
        "ta": "தாஜ்மஹால் எங்கு உள்ளது?",
        "te": "తాజ్ మహల్ ఎక్కడ ఉంది?",
    },
    "Summarization": {
        "hi": "भारतीय स्वतंत्रता संग्राम पर एक छोटा सारांश दें।",
        "ta": "இந்திய சுதந்திர போராட்டத்தின் சுருக்கம் கூறுங்கள்.",
        "te": "భారత స్వాతంత్ర్య సమరంపై సంక్షిప్తంగా వివరించండి.",
    },
    "Code Generation": {
        "hi": "दो संख्याओं को जोड़ने वाला पायथन प्रोग्राम लिखें।",
        "ta": "இரு எண்ணங்களை கூட்டும் பைத்தான் நிரலை எழுதுங்கள்.",
        "te": "రెండు సంఖ్యలను కలిపే పైథాన్ ప్రోగ్రాం రాయండి.",
    },
    "Instruction-following": {
        "hi": "चाय कैसे बनाएं, चरण दर चरण समझाएं।",
        "ta": "சாயை எப்படி தயாரிப்பது என்று படிப்படியாக கூறுங்கள்.",
        "te": "టీ ఎలా తయారు చేయాలో దశలవారీగా వివరించండి.",
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
        "Prompt": df["Prompt"].iloc[0],
        "Question Type": df["Question Type"].iloc[0],
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
    all_results = []

    for q_type, lang_prompts in PROMPT_TYPES.items():
        for lang_code, native_prompt in lang_prompts.items():
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
            print(f"\n=== {q_type} | Language: {lang_name} ({lang_code}) ===")
            run_results = []

            for run in range(REPEAT_RUNS):
                print(f"\n--- Run {run + 1} ---")

                translated_english, latency_to_english = translate_to_english(native_prompt, lang_code)
                print(f"To English: {translated_english} ({latency_to_english:.2f} ms)")

                sarvam_output_english, sarvam_latency = get_sarvam_response(translated_english, SARVAM_API_KEY)
                print(f"Sarvam: {sarvam_output_english} ({sarvam_latency:.2f} ms)")

                translated_back, latency_to_native = translate_back(sarvam_output_english, lang_code)
                print(f"Back Translation: {translated_back} ({latency_to_native:.2f} ms)")

                run_results.append({
                    "Run": run + 1,
                    "Prompt": native_prompt,
                    "Question Type": q_type,
                    "Input Language": lang_name,
                    "Prompt Length": len(native_prompt),
                    "LLM Response Length": len(sarvam_output_english),
                    "To English Latency (ms)": round(latency_to_english, 2),
                    "LLM Inference Latency (ms)": round(sarvam_latency, 2),
                    "To Native Latency (ms)": round(latency_to_native, 2)
                })

            log_to_excel(run_results)
            all_results.extend(run_results)

    print(f"\nAll prompt types and languages completed. Results saved in '{EXCEL_FILENAME}'.")
