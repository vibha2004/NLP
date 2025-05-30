import requests
import time
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

HUGGINGFACE_TOKEN = ""
API_URL = "https://api-inference.huggingface.co/models/sarvamai/sarvam-m"
HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
    "Content-Type": "application/json"
}

def translate_to_english(text):
    start_time = time.time()
    result = GoogleTranslator(source='auto', target='en').translate(text)
    latency = time.time() - start_time
    return result, latency

def translate_to_language(text, target_lang_code):
    start_time = time.time()
    result = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
    latency = time.time() - start_time
    return result, latency

def query_sarvam(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 512
        }
    }

    start_time = time.time()
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    latency = time.time() - start_time
    response.raise_for_status()

    output = response.json()
    if isinstance(output, list) and "generated_text" in output[0]:
        return output[0]["generated_text"], latency
    else:
        raise ValueError("Unexpected response from Hugging Face API")

def run_batch_tests():
    prompts = {
        "bn": "জলবায়ু পরিবর্তন কী?",
        "gu": "હવામાનમાં ફેરફાર શું છે?",
        "hi": "जलवायु परिवर्तन क्या है?",
        "kn": "ಹವಾಮಾನ ಬದಲಾವಣೆ ಎಂದರೇನು?",
        "mr": "हवामान बदल म्हणजे काय?",
        "ml": "കാലാവസ്ഥ മാറ്റം എന്താണ്?",
        "or": "ପରିବେଶ ପରିବର୍ତ୍ତନ କଣ?",
        "pa": "ਮੌਸਮ ਵਿੱਚ ਤਬਦੀਲੀ ਕੀ ਹੈ?",
        "ta": "காலநிலை மாற்றம் என்பது என்ன?",
        "te": "వాతావరణ మార్పు అంటే ఏమిటి?"
    }

    for lang_code, prompt in prompts.items():
        print(f"\n\n=== Testing Language: {lang_code.upper()} ===")
        print(f"Original Prompt: {prompt}")

        
        try:
            detected_lang = detect(prompt)
        except LangDetectException:
            detected_lang = lang_code
            print(f"Could not auto-detect language. Defaulting to: {lang_code}")

        
        english_prompt, to_en_latency = translate_to_english(prompt)
        print(f"\n→ English_1 (Translated Prompt): {english_prompt}")
        print(f"Translation to English latency: {to_en_latency:.3f} sec")

        
        print("\nSending prompt to Sarvam-M...")
        sarvam_response, llm_latency = query_sarvam(english_prompt)
        print(f"Sarvam-M inference latency: {llm_latency:.3f} sec")
        print("\n→ Sarvam-M's Response (English):")
        print(sarvam_response)

        
        final_response, back_to_lang_latency = translate_to_language(sarvam_response, detected_lang)
        print(f"\n← Translated back to {lang_code}: {final_response}")
        print(f"Back translation latency: {back_to_lang_latency:.3f} sec")

        
        english_round_trip, round_trip_latency = translate_to_english(final_response)
        print(f"\n⇌ English_2 (Round-Trip): {english_round_trip}")
        print(f"Round-trip translation to English latency: {round_trip_latency:.3f} sec")

        
        print("\n=== Comparison of English Translations ===")
        print(f"English_1: {english_prompt}")
        print(f"English_2: {english_round_trip}")
        print("==========================================")

if __name__ == "__main__":
    run_batch_tests()
