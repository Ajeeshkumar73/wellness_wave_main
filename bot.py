import os
import re
import difflib
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Strong, focused system prompt
SYSTEM_PROMPT = (
    "You are a specialized Health & Lifestyle AI for Wellness Wave. "
    "ONLY answer questions related to health, fitness, lifestyle diseases (Diabetes, BP, Heart, Obesity), and diet. "
    "If a question is NOT related to health (e.g., cooking recipes like tea, history, politics), "
    "politely reply: 'I am a health bot designed specifically for lifestyle disease and wellness support. Please ask health-related questions.' "
    "Maintain conversation context and answer follow-up questions naturally if they relate to the health discussion. "
    "DO NOT show your thinking or reasoning (<think> tags). "
    "Answer directly in 4-5 lines max. "
    "Always prioritize facts from the provided USER'S RECENT HEALTH DATA."
)

# Health scope keywords to identify valid queries early
HEALTH_KEYWORDS = [
    "diabetes", "sugar", "insulin", "glucose", "bp", "blood pressure", "heart", "cholesterol",
    "obesity", "weight", "bmi", "fat", "metabolism", "exercise", "walking", "fitness", "yoga",
    "diet", "food", "nutrition", "carbs", "protein", "vitamins", "sleep", "stress", "lifestyle",
    "thyroid", "health", "wellness", "anxiety", "risk", "precaution", "doctor", "medicine", "pill",
    "pain", "joint", "muscle", "headache", "immune", "alcohol", "drugs", "smoking", "symptom", "ill"
]

def is_health_related(text: str) -> bool:
    text = text.lower()
    
    # Check if any keyword matches
    for word in HEALTH_KEYWORDS:
        if word in text:
            return True
    
    # Basic fuzzy matching for typos
    words = re.findall(r'\b\w+\b', text)
    for w in words:
        if len(w) > 4:
            matches = difflib.get_close_matches(w, HEALTH_KEYWORDS, n=1, cutoff=0.7)
            if matches:
                return True
                
    # Context-aware follow-up check: If the word is very short might be a follow-up
    # like "why?", "is it?", "how?"
    follow_ups = ["why", "how", "is it", "what", "tell", "explain", "good", "bad", "no", "yes"]
    if any(text.strip().startswith(f) for f in follow_ups) and len(text.split()) < 4:
        return True

    return False

def clean_response(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"[#*_`]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_max_tokens(user_input: str) -> int:
    if len(user_input) < 80:
        return 150
    return 300

def lifestyle_disease_chat(user_input, context="", history=None):
    # 🚫 Strict health-bot guardrail
    if not is_health_related(user_input) and not (history and len(history) > 0):
        return "I am a health bot designed specifically for lifestyle disease and wellness support. Please ask health-related questions."

    # Construct message list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "system", 
            "content": f"USER'S RECENT HEALTH DATA:\n{context}\n\nPrimary source for user query interpretation."
        })

    if history:
        for msg in history[-10:]: 
            messages.append(msg)

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=0.4,
            max_tokens=get_max_tokens(user_input)
        )

        reply = response.choices[0].message.content
        return clean_response(reply)
    except Exception as e:
        print(f"Chat error: {e}")
        return "I'm having trouble connecting right now. Please try again soon."
