import requests

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
    response.raise_for_status()  
    return response.json()["choices"][0]["message"]["content"]

def main():
    print("Mistral CLI: Let the model translate your input and respond in English\n")
    user_input = input("Enter your prompt in any language (untranslated): ")

    mistral_prompt = (
        "Please translate the following user input to English, then answer the question "
        "in English. At the end, provide both the translated question and your answer.\n\n"
        f"User input: {user_input}\n\n"
        "Format:\nTranslated Question: <translated text>\nAnswer: <response in English>"
    )

    print("\nSending prompt to Mistral...")
    english_response = query_mistral(mistral_prompt)

    translated_question = ""
    english_answer = ""
    for line in english_response.splitlines():
        if line.startswith("Translated Question:"):
            translated_question = line[len("Translated Question:"):].strip()
        elif line.startswith("Answer:"):
            english_answer = line[len("Answer:"):].strip()

    
    translate_back_prompt = (
        f"Please translate the following answer back to the original language of this input:\n\n"
        f"Original Input: {user_input}\n"
        f"English Answer: {english_answer}\n\n"
        f"Just give the translated answer in the original language."
    )

    translated_back = query_mistral(translate_back_prompt)

    print("\nMistral's Response:")
    print("----------------------------")
    print(f"Translated Question: {translated_question}")
    print(f"Answer in English: {english_answer}")
    print(f"Answer in Original Language: {translated_back}")
    print("----------------------------")

if __name__ == "__main__":
    main()
