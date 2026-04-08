import os
from openai import OpenAI

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(prompt: str):
    print("START")
    print("STEP: Sending request to model")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    print("STEP: Received response")
    print("END")

    return response.choices[0].message.content


if __name__ == "__main__":
    result = run_inference("Test prompt")
    print(result)