import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)

models_to_test = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

for model_name in models_to_test:

    print("\n" + "=" * 50)
    print("Testing:", model_name)
    print("=" * 50)

    try:

        response = client.models.generate_content(
            model=model_name,
            contents="Say hello in one sentence."
        )

        print("SUCCESS")
        print(response.text)

    except Exception as e:

        print("FAILED")
        print(e)