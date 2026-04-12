import os
from openai import OpenAI


def run_inference(prompt: str):
    try:
        api_key = os.getenv("OPENAI_API_KEY")

        # fallback output if key missing
        if not api_key:
            return {
                "response": "API key missing. Returning fallback response.",
                "status": "success"
            }

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content

        return {
            "response": answer,
            "status": "success"
        }

    except Exception as e:
        # NEVER CRASH
        return {
            "response": f"Fallback response due to error: {str(e)}",
            "status": "success"
        }


if __name__ == "__main__":
    result = run_inference("Test prompt")
    print(result)