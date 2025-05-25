import ollama

SYSTEM_PROMPT = """
You are a toxicity classifier trained to identify different types of toxic content. Given a text, predict the probability of it belonging to each of these categories:
- toxic: general offensive or harmful language.
- severe_toxic: extremely harmful or aggressive language.
- obscene: vulgar or inappropriate language.
- threat: language indicating potential harm or danger.
- insult: derogatory or belittling remarks.
- identity_hate: hate speech targeting specific groups or identities.

Return a JSON object like this:
{
  "toxic": [0-1],
  "severe_toxic": [0-1],
  "obscene": [0-1],
  "threat": [0-1],
  "insult": [0-1],
  "identity_hate": [0-1]
}

Then analyze the JSON as follows:

- If 'threat' > 0.5 → Threat: Harmful Instructions, Safety: Unsafe
- Else if 'severe_toxic' > 0.5 or 'identity_hate' > 0.5 → Threat: Toxic Content, Safety: Unsafe
- Else if the input contains self-harm keywords like "hurt yourself", "kill yourself", "end your life", "I want to die", etc. → Threat: Self-Harm Promotion, Safety: Unsafe
- Else if any category > 0.5 → Threat: None, Safety: Unsafe
- Else → Threat: None, Safety: Safe

Then append a summary string:
"Summary: Safety: [Safe/Unsafe], Threat: [None/Toxic Content/Harmful Instructions/Self-Harm Promotion]"
"""


def classify_prompt(text):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text to analyze: {text}"}
        ]
    )
    return response['message']['content']


def generate_response(user_input):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Provide a concise response to the user input."},
            {"role": "user", "content": user_input}
        ]
    )
    return response['message']['content']


def guardrail_system():
    while True:
        user_input = input()
        if user_input.lower() == "exit":
            break

        input_result = classify_prompt(user_input)
        print(input_result)

        output_response = generate_response(user_input)
        output_result = classify_prompt(output_response)
        print(output_response + '\n\n')
        print(output_result)


if __name__ == "__main__":
    guardrail_system()
