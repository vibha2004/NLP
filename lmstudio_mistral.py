import requests
import time
from deep_translator import GoogleTranslator
from langdetect import detect

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

def query_mistral(prompt):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    latency = time.time() - start_time

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"], latency

def main():
    print("Mistral CLI with Auto Translation & Latency Metrics\n")
    user_input = input("Enter your prompt in any language: ")

    detected_lang = detect(user_input)
    print(f"\nDetected language code: {detected_lang}")

    trans_to_en_latency = 0
    if detected_lang != "en":
        english_prompt, trans_to_en_latency = translate_to_english(user_input)
    else:
        english_prompt = user_input

    print(f"\nTranslated to English:\n{english_prompt}")
    if detected_lang != "en":
        print(f"Translation to English latency: {trans_to_en_latency:.3f} seconds")

    print("\nSending prompt to Mistral...")
    response, llm_latency = query_mistral(english_prompt)

    print(f"\nMistral LLM inference latency: {llm_latency:.3f} seconds")
    print("\nMistral's Response (English):")
    print("----------------------------")
    print(response)
    print("----------------------------")

    if detected_lang != "en":
        translated_response, trans_back_latency = translate_to_language(response, detected_lang)
        print(f"\nTranslation back to '{detected_lang}' latency: {trans_back_latency:.3f} seconds")
        print("\nFinal Translated Response:")
        print("----------------------------")
        print(translated_response)
        print("----------------------------")

if __name__ == "__main__":
    main()
