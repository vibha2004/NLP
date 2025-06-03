import time
import requests
import pandas as pd
from typing import Optional, List, Dict
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException


HF_TOKEN = ""
REPEAT_RUNS = 3 
EXCEL_FILENAME = "mistral_latency_metrics.xlsx"
LANGUAGE_MAP = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'ml': 'Malayalam', 'kn': 'Kannada', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'or': 'Odia', 'en': 'English'
}

PROMPT_TYPES = {
    "Q&A": {
        "hi": "ताज महल कहाँ है?",
        "ta": "தாஜ்மஹால் எங்கு உள்ளது?",
        "te": "తాజ్ మహల్ ఎక్కడ ఉంది?",
    },
    "Summarization": {
        "hi": "स्वतंत्रता संग्राम का संक्षेप में वर्णन करें।",
        "ta": "இந்திய சுதந்திரப் போராட்டத்தின் சுருக்கம் என்ன?",
        "te": "భారత స్వాతంత్ర్య సమరంపై సంక్షిప్త వివరణ ఇవ్వండి.",
    },
    "Code Generation": {
        "hi": "दो संख्याओं को जोड़ने वाला पायथन प्रोग्राम लिखें।",
        "ta": "இரு எண்களை கூட்டும் பைதான் நிரலை எழுதுங்கள்.",
        "te": "రెండు సంఖ్యలను కలిపే పైథాన్ ప్రోగ్రామ్ రాయండి.",
    },
    "Instruction-following": {
        "hi": "चाय कैसे बनाएं? चरण दर चरण समझाएं।",
        "ta": "சாயை எப்படி தயாரிப்பது? படிப்படியாக விளக்கவும்.",
        "te": "టీ ఎలా తయారు చేయాలి? దశలవారీగా వివరించండి.",
    }
}


def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None


def translate_to_english(text: str, source_lang: str) -> str:
    return GoogleTranslator(source=source_lang, target='en').translate(text)


def translate_back(text: str, target_lang: str) -> str:
    max_chunk_size = 4500
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    translated_parts = [
        GoogleTranslator(source='en', target=target_lang).translate(chunk)
        for chunk in chunks
    ]
    return ' '.join(translated_parts)

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

# Core run function
def run_experiment(prompt: str, lang_code: str, q_type: str) -> List[Dict]:
    lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
    records = []

    for run in range(REPEAT_RUNS):
        print(f"\nRun {run + 1} | {q_type} | {lang_name}")

        start = time.time()
        translated_english = translate_to_english(prompt, lang_code)
        latency_to_english = (time.time() - start) * 1000

        start = time.time()
        mistral_output = get_llm_response_from_api(translated_english, HF_TOKEN)
        mistral_latency = (time.time() - start) * 1000

        start = time.time()
        translated_back = translate_back(mistral_output, lang_code)
        latency_to_native = (time.time() - start) * 1000

        print(f"Prompt (EN): {translated_english}")
        print(f"Output (EN): {mistral_output}")
        print(f"Back-translated: {translated_back}")

        records.append({
            "Run": run + 1,
            "Prompt": prompt,
            "Question Type": q_type,
            "Input Language": lang_name,
            "Prompt Length": len(prompt),
            "LLM Response Length": len(mistral_output),
            "To English Latency (ms)": round(latency_to_english, 2),
            "LLM Inference Latency (ms)": round(mistral_latency, 2),
            "To Native Latency (ms)": round(latency_to_native, 2)
        })

    return records

# Save to Excel with mean
def save_results_with_mean(results: List[Dict], file_name: str):
    df = pd.DataFrame(results)

    mean_row = {
        "Run": "Mean",
        "Prompt": df["Prompt"].iloc[0],
        "Question Type": df["Question Type"].iloc[0],
        "Input Language": df["Input Language"].iloc[0],
        "Prompt Length": round(df["Prompt Length"].mean(), 2),
        "LLM Response Length": round(df["LLM Response Length"].mean(), 2),
        "To English Latency (ms)": round(df["To English Latency (ms)"].mean(), 2),
        "LLM Inference Latency (ms)": round(df["LLM Inference Latency (ms)"].mean(), 2),
        "To Native Latency (ms)": round(df["To Native Latency (ms)"].mean(), 2),
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    if EXCEL_FILENAME in file_name:
        try:
            old = pd.read_excel(file_name)
            df = pd.concat([old, df], ignore_index=True)
        except:
            pass

    df.to_excel(file_name, index=False)
    print(f"Results saved to {file_name}")

# Main entry
if __name__ == "__main__":
    all_results = []

    for q_type, prompts in PROMPT_TYPES.items():
        for lang_code, prompt in prompts.items():
            records = run_experiment(prompt, lang_code, q_type)
            save_results_with_mean(records, EXCEL_FILENAME)
            all_results.extend(records)

    print(f"\nAll prompt types and languages completed. Results saved in '{EXCEL_FILENAME}'.")
