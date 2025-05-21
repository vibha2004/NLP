from typing import Optional, Tuple
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LANGUAGE_MAP = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'ml': 'Malayalam', 'kn': 'Kannada', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'ur': 'Urdu', 'or': 'Odia', 'as': 'Assamese', 'en': 'English'
}

def translate_indian_to_english(text: str) -> Optional[Tuple[str, str]]:
    try:
        print("Translating text to English...")
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx', 'sl': 'auto', 'tl': 'en', 'dt': 't', 'q': text
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        translated_text = result[0][0][0] if result and result[0] and result[0][0] else None
        detected_lang_code = result[2] if len(result) > 2 else 'unknown'
        return translated_text, detected_lang_code
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return None

def load_model(model_name: str):
    try:
        print("Loading Falcon model. This may take a while...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  
            torch_dtype=torch.float32  
        )
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def get_llm_response(prompt: str, tokenizer, model) -> str:
    try:
        print("Generating response from Falcon LLM...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

if __name__ == "__main__":
    user_text = input("Enter text in any Indian language: ")

    translation_result = translate_indian_to_english(user_text)

    if translation_result:
        translated_text, detected_lang = translation_result
        lang_name = LANGUAGE_MAP.get(detected_lang, detected_lang)
        print(f"\nDetected Language: {lang_name} ({detected_lang})")
        print(f"Translated to English: {translated_text}")

        model_name = "tiiuae/falcon-rw-1b"
        tokenizer, model = load_model(model_name)

        if tokenizer and model:
            response = get_llm_response(translated_text, tokenizer, model)
            print(f"\nLLM Response:\n{response}")
        else:
            print("LLM model loading failed. Please check system memory or GPU availability.")
    else:
        print("Translation failed.")


