import requests
from deep_translator import GoogleTranslator
from langdetect import detect

def translate_to_english(text):
    
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_language(text, target_lang_code):
    
    return GoogleTranslator(source='auto', target=target_lang_code).translate(text)

def query_mistral(prompt):
    
    url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise error for bad responses
    return response.json()["choices"][0]["message"]["content"]

def main():
    print("Mistral CLI with Auto Translation\n")
    user_input = input("Enter your prompt in any language: ")

    
    detected_lang = detect(user_input)
    print(f"\nDetected language code: {detected_lang}")

    
    if detected_lang != "en":
        english_prompt = translate_to_english(user_input)
    else:
        english_prompt = user_input
    print(f"\nTranslated to English:\n{english_prompt}")

   
    print("\nSending prompt to Mistral...")
    response = query_mistral(english_prompt)

    
    print("\nMistral's Response (English):")
    print("----------------------------")
    print(response)
    print("----------------------------")

    
    if detected_lang != "en":
        translated_response = translate_to_language(response, detected_lang)
        print(f"\nTranslated back to your language ({detected_lang}):")
        print("----------------------------")
        print(translated_response)
        print("----------------------------")

if __name__ == "__main__":
    main()

