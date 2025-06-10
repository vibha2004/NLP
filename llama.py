import requests
import time
from typing import Optional
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import pandas as pd
import os
from rouge_score import rouge_scorer

# Constants
HF_API_TOKEN = ""
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
REPEAT_RUNS = 3
EXCEL_FILENAME = "llama_latency_metrics_with_cost.xlsx"

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
        print("⚠️ JSON decoding error")
        print("Raw Response:", response.text)
        return "[Error parsing response]", (end - start) * 1000

    if response.status_code != 200:
        print(f"⚠️ API returned status {response.status_code}: {result}")
        return f"[Error {response.status_code}: {result.get('error', response.text)}]", (end - start) * 1000

    try:
        generated_text = result[0]['generated_text']
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text, (end - start) * 1000
    except (KeyError, IndexError, TypeError):
        print("⚠️ Unexpected response format:", result)
        return "[Unexpected response format]", (end - start) * 1000


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(input_text: str, output_text: str):
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text)
    input_cost = (input_tokens / 1000) * 0.0015
    output_cost = (output_tokens / 1000) * 0.0020
    return input_tokens, output_tokens, round(input_cost + output_cost, 6)


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
        "LLM Inference Latency (ms)": round(df["LLM Inference Latency (ms)"].mean(), 2),
        "To Native Latency (ms)": round(df["To Native Latency (ms)"].mean(), 2),
        "Input Tokens": round(df["Input Tokens"].mean(), 2),
        "Output Tokens": round(df["Output Tokens"].mean(), 2),
        "Estimated Cost ($)": round(df["Estimated Cost ($)"].mean(), 6),
        "ROUGE-1 F1": round(df.get("ROUGE-1 F1", pd.Series([0])).mean(), 4),
        "ROUGE-2 F1": round(df.get("ROUGE-2 F1", pd.Series([0])).mean(), 4),
        "ROUGE-L F1": round(df.get("ROUGE-L F1", pd.Series([0])).mean(), 4)
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(filename, index=False)


if __name__ == "__main__":
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_results = []

    for q_type, lang_prompts in PROMPT_TYPES.items():
        for lang_code, native_prompt in lang_prompts.items():
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
            print(f"\n=== {q_type} | Language: {lang_name} ({lang_code}) ===")
            run_results = []

            for run in range(REPEAT_RUNS):
                print(f"\n--- Run {run + 1} ---")

                translated_english, latency_to_english = translate_to_english(native_prompt, lang_code)
                hf_output, hf_latency = get_hf_response(translated_english, HF_API_TOKEN)
                translated_back, latency_to_native = translate_back(hf_output, lang_code)

                input_tokens, output_tokens, cost = estimate_cost(translated_english, hf_output)

                print(f"Prompt (Native): {native_prompt}")
                print(f"Prompt (English): {translated_english}")
                print(f"LLM Output (English): {hf_output}")
                print(f"Translated Back to {lang_name}: {translated_back}")
                print(f"Tokens - Input: {input_tokens}, Output: {output_tokens}, Cost: ${cost:.6f}")

                result = {
                    "Run": run + 1,
                    "Prompt": native_prompt,
                    "Question Type": q_type,
                    "Input Language": lang_name,
                    "Prompt Length": len(native_prompt),
                    "LLM Response Length": len(hf_output),
                    "To English Latency (ms)": round(latency_to_english, 2),
                    "LLM Inference Latency (ms)": round(hf_latency, 2),
                    "To Native Latency (ms)": round(latency_to_native, 2),
                    "Input Tokens": input_tokens,
                    "Output Tokens": output_tokens,
                    "Estimated Cost ($)": cost
                }

                if q_type == "Summarization":
                    scores = scorer.score(native_prompt, translated_back)
                    result["ROUGE-1 F1"] = round(scores['rouge1'].fmeasure, 4)
                    result["ROUGE-2 F1"] = round(scores['rouge2'].fmeasure, 4)
                    result["ROUGE-L F1"] = round(scores['rougeL'].fmeasure, 4)
                    print(f"ROUGE-1: {result['ROUGE-1 F1']}, ROUGE-2: {result['ROUGE-2 F1']}, ROUGE-L: {result['ROUGE-L F1']}")

                run_results.append(result)

            log_to_excel(run_results)
            all_results.extend(run_results)
            print(f"\nCompleted {q_type} in {lang_name}. Logged {len(run_results)} runs.\n")

    print(f"\n All prompt types and languages completed. Results saved in '{EXCEL_FILENAME}'.")



