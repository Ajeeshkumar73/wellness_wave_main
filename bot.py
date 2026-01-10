import os
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Strong, focused system prompt
SYSTEM_PROMPT = (
     "You are a healthcare chatbot. "
    "DO NOT show your thinking or reasoning. "
    "DO NOT use <think> or explanations of your thought process. "
    "Answer directly. "
    "Keep the reply SHORT and COMPLETE. "
    "Use bullet points ONLY if the question asks for tips or benefits. "
    "Limit the answer to 4–5 lines maximum. "
    "Do NOT diagnose diseases. "
    "Suggest consulting a doctor briefly if relevant."
)

# Health scope keywords
HEALTH_KEYWORDS = [
    "diabetes", "blood sugar", "sugar level", "glucose",
    "bp", "blood pressure", "hypertension",
    "heart", "cholesterol",
    "obesity", "weight", "weight loss",
    "exercise", "walking", "workout", "yoga",
    "diet", "food", "nutrition",
    "sleep", "stress", "lifestyle", "thyroid"
]

def is_health_related(text: str) -> bool:
    text = text.lower()

    # direct keyword match
    for word in HEALTH_KEYWORDS:
        if word in text:
            return True

    # fallback: common health intent words
    health_intent_words = ["reduce", "control", "prevent", "manage", "improve"]
    if any(w in text for w in health_intent_words):
        return True

    return False


def clean_response(text: str) -> str:
    # Remove <think> blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)

    # Remove markdown symbols
    text = re.sub(r"[#*_`]", "", text)

    # Clean extra new lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def get_max_tokens(user_input: str) -> int:
    # Keep answers short by design
    if len(user_input) < 80:
        return 150
   

def lifestyle_disease_chat(user_input: str) -> str:
    # 🚫 Block non-health questions
    if not is_health_related(user_input):
        return (
            "I can help only with lifestyle disease related doubts 😊 "
            "Please ask about diet, exercise, diabetes, BP, weight, stress, or heart health."
        )

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0.4,
        max_tokens=get_max_tokens(user_input)
    )

    reply = response.choices[0].message.content
    return clean_response(reply)
