"""
RetailNext Intelligent Search - Streamlit Demo
OpenAI Agents SDK + vector search + multimodal rerank + virtual try-on.

Quick Orientation
-----------------
This app is a conversational retail demo that:
1) Collects user intent (category + color) and optional event context
2) Runs MongoDB Atlas vector search (top 5)
3) Reranks with multimodal reasoning (event-first)
4) Presents top 3 results with explanations
5) Supports add-to-cart, stock check, and a virtual try-on flow

Core Flow (High Level)
----------------------
User message -> intent extraction -> optional event confirm ->
vector search -> rerank -> summary + UI cards -> quick actions.

Key Session State
-----------------
- user_profile: selected demo user from MongoDB
- event_context: event from user input or confirmed from profile
- last_initial_results / last_reranked_results: most recent results
- cart: simple in-session cart
- tryon_*: virtual try-on selection + output
"""

import os
import json
import re
import time
import base64
import mimetypes
import urllib.request
import ssl
from datetime import datetime
from uuid import uuid4

import certifi
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

from agents import Agent, Runner, ModelSettings

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "retail_demo"
COLLECTION_NAME = "productAndEmbeddings"
USER_COLLECTION = "user_profiles"
CONVO_COLLECTION = "user_conversation_memory"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
INTENT_MODEL = "gpt-4o-mini"

APP_TITLE = "RetailNext Intelligent Search"
APP_ICON = "🛍️"

WEDDING_KEYWORDS = [
    "wedding", "ceremony", "reception", "bridal", "bridesmaid", "groom"
]

EVENT_KEYWORDS = [
    "wedding", "ceremony", "reception", "bridal", "bridesmaid", "groom",
    "party", "birthday", "anniversary", "interview", "date", "office",
    "vacation", "beach", "garden",
    "business trip", "work trip", "business travel", "conference"
]

CATEGORY_KEYWORDS = {
    "t-shirt": "Tshirts",
    "t shirt": "Tshirts",
    "tshirt": "Tshirts",
    "dress": "Dresses",
    "dresses": "Dresses",
    "gown": "Dresses",
    "shirt": "Shirts",
    "shirts": "Shirts",
    "top": "Tops",
    "tops": "Tops",
    "blouse": "Tops",
    "heels": "Heels",
    "flats": "Flats",
    "sandals": "Sandals",
    "shoes": "Shoes",
    "pants": "Trousers",
    "trousers": "Trousers",
    "jeans": "Jeans",
    "skirt": "Skirts",
    "kurta": "Kurtas",
    "saree": "Sarees",
    "sari": "Sarees",
}

COLOR_KEYWORDS = [
    "navy blue", "off white", "dark blue", "light blue",
    "black", "white", "red", "blue", "navy", "pink", "green", "beige",
    "cream", "orange", "yellow", "purple", "grey", "gray", "brown",
    "maroon", "gold", "silver", "multi"
]

MEN_WORDS = ["men", "men's", "mens", "male", "for men", "for my husband", "for him", "groom"]
WOMEN_WORDS = ["women", "women's", "womens", "female", "for women", "for my wife", "for her", "bride", "bridesmaid"]

CHECK_STOCK_WORDS = ["check stock", "stock", "availability", "available"]
ADD_TO_CART_WORDS = ["add to cart", "add it", "add this", "yes", "yep", "sure"]
CHECKOUT_WORDS = ["checkout", "pay", "purchase", "buy now", "place order"]

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CLIENTS
# =============================================================================

@st.cache_resource
def get_mongodb_client():
    if not MONGODB_URI:
        st.error("MONGODB_URI not found in environment variables")
        return None
    return MongoClient(MONGODB_URI, tlsCAFile=certifi.where())


@st.cache_resource
def get_openai_client():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment variables")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


mongo_client = get_mongodb_client()
openai_client = get_openai_client()

if mongo_client:
    db = mongo_client[DATABASE_NAME]
else:
    db = None

# =============================================================================
# DATA ACCESS
# =============================================================================

def get_all_users():
    if db is None:
        return []
    return list(db[USER_COLLECTION].find({}, {"_id": 1, "name": 1, "email": 1}))


def get_user_profile(user_id):
    if db is None or not user_id:
        return None
    return db[USER_COLLECTION].find_one({"_id": user_id})


def get_user_conversations(user_id, limit: int = 3):
    if db is None or not user_id:
        return []
    return list(
        db[CONVO_COLLECTION]
        .find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )


def get_last_conversation_context(user_id, max_messages: int = 4) -> str:
    """Return a short text context from the most recent conversation."""
    conversations = get_user_conversations(user_id, limit=1)
    if not conversations:
        return ""
    messages = conversations[0].get("messages", []) or []
    tail = messages[-max_messages:]
    lines = []
    for m in tail:
        role = m.get("role", "user")
        content = m.get("content", "")
        if len(content) > 120:
            content = content[:120] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def save_conversation(user_id, session_id, messages):
    if db is None or not user_id or not session_id:
        return
    now = datetime.now().isoformat()
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages,
        "created_at": messages[0]["timestamp"] if messages else now,
        "updated_at": messages[-1]["timestamp"] if messages else now,
    }
    db[CONVO_COLLECTION].update_one(
        {"session_id": session_id},
        {"$set": doc},
        upsert=True
    )

# =============================================================================
# INTENT HELPERS
# =============================================================================

def detect_wedding_intent(text: str) -> bool:
    text_lower = text.lower()
    return any(k in text_lower for k in WEDDING_KEYWORDS)


def extract_event_type(text: str) -> str:
    text_lower = text.lower()
    for key in EVENT_KEYWORDS:
        if key in text_lower:
            if key in ["bridal", "bridesmaid"]:
                return "wedding"
            if key == "groom":
                return "wedding"
            if key in ["business trip", "work trip", "business travel", "conference"]:
                return "business_trip"
            return key
    return ""


def infer_intent_llm(user_text: str, last_context: str) -> dict:
    """
    Use a small OpenAI model to extract intent.
    Returns: {category, color, event_type, gender_override}
    """
    if openai_client is None:
        return {}

    prompt = (
        "Extract the product intent from the user's message. "
        "Return ONLY JSON with keys: category, color, event_type, gender_override. "
        "category: short product type (e.g., Tshirts, Shirts, Dresses, Jeans, Shoes, Heels). "
        "color: color name if mentioned, else empty string. "
        "event_type: occasion if mentioned (wedding, business_trip, party, interview, date, vacation, beach, garden, office, conference), else empty string. "
        "gender_override: Men/Women only if the user explicitly asks for that gender, else empty string. "
        "If context helps, use it, but do not guess."
    )

    payload = {
        "last_conversation_context": last_context,
        "user_message": user_text
    }

    try:
        response = openai_client.responses.create(
            model=INTENT_MODEL,
            input=[{"role": "user", "content": f"{prompt}\n\n{json.dumps(payload, indent=2)}"}],
            temperature=0.0,
        )
    except Exception:
        return {}

    parsed = _extract_json(_response_text(response))
    return parsed if isinstance(parsed, dict) else {}


def extract_category(text: str) -> str:
    text_lower = text.lower()
    # Match longer keys first
    for key in sorted(CATEGORY_KEYWORDS.keys(), key=len, reverse=True):
        if key in text_lower:
            return CATEGORY_KEYWORDS[key]
    return ""


def extract_color(text: str) -> str:
    text_lower = text.lower()
    for color in sorted(COLOR_KEYWORDS, key=len, reverse=True):
        if color in text_lower:
            return color.title() if color != "multi" else "Multi"
    return ""


def detect_gender_override(text: str) -> str | None:
    text_lower = text.lower()
    for w in MEN_WORDS:
        if w in text_lower:
            return "Men"
    for w in WOMEN_WORDS:
        if w in text_lower:
            return "Women"
    return None


def wants_check_stock(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in CHECK_STOCK_WORDS)


def wants_add_to_cart(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in ADD_TO_CART_WORDS)


def wants_checkout(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in CHECKOUT_WORDS)


def build_search_query(category: str, color: str, fallback: str) -> str:
    parts = []
    if color:
        parts.append(color)
    if category:
        parts.append(category)
    if parts:
        return " ".join(parts)
    return fallback.strip()


def get_gender_filter(user_profile: dict | None, gender_override: str | None) -> list | None:
    if gender_override:
        return [gender_override, "Unisex"]
    if not user_profile:
        return None
    gender = user_profile.get("gender") or user_profile.get("gender_preference")
    if gender in ["Women", "Men"]:
        return [gender, "Unisex"]
    if gender == "Unisex":
        return ["Unisex"]
    return None

# =============================================================================
# VECTOR SEARCH
# =============================================================================

def generate_embedding(text: str) -> list:
    if openai_client is None:
        return []
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def vector_search_products(query_embedding: list, limit: int = 5, gender_filter: list | None = None) -> list:
    if db is None:
        return []

    vector_stage = {
        "index": "vector_index",
        "path": "embeddings",
        "queryVector": query_embedding,
        "numCandidates": 100,
        "limit": limit,
    }
    if gender_filter:
        vector_stage["filter"] = {"gender": {"$in": gender_filter}}

    pipeline = [
        {"$vectorSearch": vector_stage},
        {
            "$project": {
                "_id": 1,
                "id": 1,
                "productDisplayName": 1,
                "price": 1,
                "stock_quantity": 1,
                "baseColour": 1,
                "articleType": 1,
                "image": 1,
                "image_url": 1,
                "imageUrl": 1,
                "gender": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    try:
        return list(db[COLLECTION_NAME].aggregate(pipeline))
    except Exception as exc:
        st.error(f"Vector search error: {exc}")
        return []


def format_products_for_ui(results: list) -> list:
    formatted = []
    for product in results:
        image_url = (
            product.get("image_url")
            or product.get("imageUrl")
            or product.get("image")
        )
        formatted.append({
            "id": str(product.get("_id", "")),
            "name": product.get("productDisplayName", "Unknown"),
            "price": product.get("price", 0),
            "stock_quantity": product.get("stock_quantity", None),
            "color": product.get("baseColour", ""),
            "type": product.get("articleType", ""),
            "gender": product.get("gender", ""),
            "vector_score": round(product.get("score", 0), 3),
            "image_url": image_url
        })
    return formatted

# =============================================================================
# MULTIMODAL RERANK
# =============================================================================

def _response_text(resp) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # Fallback: try to reconstruct from output items
    try:
        texts = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        texts.append(c.text)
        return "\n".join(texts)
    except Exception:
        return ""


def _extract_json(text: str):
    if not text:
        return None

    # Strip markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract JSON array (for lists)
    match = re.search(r'\[[\s\S]*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Try to extract JSON object (for dicts)
    match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def rerank_products_multimodal(products: list, user_profile: dict, event: dict, user_query: str) -> list:
    if openai_client is None or not products:
        return []

    prefs = user_profile.get("preferences", {}) if user_profile else {}
    style_prefs = ", ".join(prefs.get("style_preferences", [])) or "N/A"
    colors = ", ".join(prefs.get("favorite_colors", [])) or "N/A"
    price_range = prefs.get("price_range", {})
    budget = f"${price_range.get('min', 0)}-${price_range.get('max', 0)}" if price_range else "N/A"

    event_text = "No upcoming event."
    if event:
        event_text = (
            f"Event: {event.get('event_type', 'Event')}; "
            f"Date: {event.get('date', '')}; "
            f"Dress code: {event.get('dress_code', '')}; "
            f"Venue: {event.get('venue', '')}; "
            f"Notes: {event.get('notes', '')}"
        )

    prompt = (
        "You are reranking product results for a fashion shopping assistant. "
        "Use the event details as higher priority than personal preferences. "
        "Score each product with event_score and preference_score between 0 and 1. "
        "Return JSON array of objects: {id, event_score, preference_score, reasoning}. "
        "Reasoning must be 1-2 short bullet points. "
        "If image is missing, lower the event_score and mention it.\n\n"
        f"User query: {user_query}\n"
        f"User preferences: styles={style_prefs}; colors={colors}; budget={budget}.\n"
        f"{event_text}\n"
    )

    # Build content for vision API
    content = [{"type": "text", "text": prompt}]
    for idx, p in enumerate(products, start=1):
        product_text = (
            f"Product {idx}: id={p['id']}; name={p['name']}; color={p['color']}; "
            f"type={p['type']}; price=${p['price']:.2f}; gender={p.get('gender','')};"
        )
        content.append({"type": "text", "text": product_text})

        if p.get("image_url"):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": p["image_url"],
                    "detail": "low"
                }
            })
        else:
            content.append({"type": "text", "text": "Image: not available."})

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"Rerank API error: {e}")
        return []

    data = _extract_json(response_text)
    if not isinstance(data, list):
        return []

    by_id = {}
    for item in data:
        pid = str(item.get("id", ""))
        if not pid:
            continue
        event_score = float(item.get("event_score", 0))
        pref_score = float(item.get("preference_score", 0))
        overall = round(0.6 * event_score + 0.4 * pref_score, 3)
        reasoning = item.get("reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        by_id[pid] = {
            "event_score": max(0.0, min(1.0, event_score)),
            "preference_score": max(0.0, min(1.0, pref_score)),
            "overall_score": max(0.0, min(1.0, overall)),
            "reasoning": reasoning,
        }

    reranked = []
    for p in products:
        scores = by_id.get(p["id"])
        if not scores:
            continue
        merged = {**p, **scores}
        reranked.append(merged)

    reranked.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    return reranked[:3]


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str | None) -> tuple[str, int, str]:
    mime = mime_type or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}", len(image_bytes), mime


def parse_data_url(data_url: str) -> tuple[bytes | None, str | None]:
    try:
        header, b64 = data_url.split(",", 1)
        mime = header.split(";")[0].replace("data:", "")
        return base64.b64decode(b64), mime
    except Exception:
        return None, None


def fetch_image_bytes(url: str) -> tuple[bytes | None, int | None, str | None, str | None]:
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context) as resp:
            content_type = resp.headers.get("Content-Type", "")
            image_bytes = resp.read()
        mime_type = content_type.split(";")[0] if content_type else None
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(url)
        return image_bytes, len(image_bytes), mime_type, None
    except Exception as exc:
        return None, None, None, str(exc)




def generate_tryon_image(user_image_input: dict, product_image_url: str) -> tuple[str | None, dict]:
    """
    Generate a virtual try-on image using the image_generation tool.
    user_image_input: {"type": "input_image", "image_url": ...} or {"type": "input_image", "image_base64": ...}
    Returns base64 image string or None.
    """
    debug = {"user_image": {}, "product_image": {}, "error": None}
    if openai_client is None:
        debug["error"] = "OpenAI client not initialized."
        return None, debug
    if not product_image_url:
        debug["error"] = "Product image URL missing."
        return None, debug

    prompt = (
        "Edit the first image (the person) so they are wearing the clothing item from the second image. "
        "Preserve the person's face, identity, pose, body shape, and background. Do not change the face of the person. "
        "Make the clothing fit naturally and look realistic."
    )

    # Load user image bytes
    user_bytes = None
    user_mime = None
    if user_image_input.get("bytes"):
        user_bytes = user_image_input.get("bytes")
        user_mime = user_image_input.get("mime")
        debug["user_image"] = user_image_input.get("debug", {})
    elif user_image_input.get("url"):
        user_bytes, size, user_mime, err = fetch_image_bytes(user_image_input.get("url"))
        debug["user_image"] = {"source": "url", "bytes": size, "mime": user_mime, "error": err}
    elif user_image_input.get("data_url"):
        user_bytes, user_mime = parse_data_url(user_image_input.get("data_url"))
        debug["user_image"] = {"source": "data_url", "bytes": len(user_bytes or b""), "mime": user_mime}

    if not user_bytes:
        debug["error"] = "Could not load user image."
        return None, debug

    # Load product image bytes
    product_bytes = None
    product_mime = None
    if product_image_url.startswith("data:"):
        product_bytes, product_mime = parse_data_url(product_image_url)
        debug["product_image"] = {"source": "data_url", "bytes": len(product_bytes or b""), "mime": product_mime}
    else:
        product_bytes, size, product_mime, err = fetch_image_bytes(product_image_url)
        debug["product_image"] = {"source": "url", "bytes": size, "mime": product_mime, "error": err}

    if not product_bytes:
        debug["error"] = "Could not load product image."
        return None, debug

    try:
        user_mime = user_mime or "image/jpeg"
        product_mime = product_mime or "image/jpeg"
        user_name = f"user.{user_mime.split('/')[-1]}"
        product_name = f"product.{product_mime.split('/')[-1]}"
        result = openai_client.images.edit(
            model="gpt-image-1.5",
            image=[(user_name, user_bytes, user_mime), (product_name, product_bytes, product_mime)],
            prompt=prompt,
            size="1024x1024",
            quality="high",
        )
        image_base64 = result.data[0].b64_json if result.data else None
        if not image_base64:
            debug["error"] = "No image returned by Images API."
            return None, debug
        return image_base64, debug
    except Exception as exc:
        debug["error"] = f"OpenAI error: {exc}"
        return None, debug

# =============================================================================
# AGENT SETUP (summary only)
# =============================================================================

def build_agent():
    instructions = (
        "You are a warm, empathetic shopping assistant. Summarize the reranked top 3 results "
        "in a friendly, conversational tone. Mention event fit if an upcoming event exists. "
        "Provide 1-2 extra recommendations if available from the initial results. "
        "Format:\n"
        "1) Short friendly intro.\n"
        "2) Bullet list of top 3 with concise reasons.\n"
        "3) Optional extra recommendations section.\n"
        "4) Friendly closing question: ask if they want to add to cart and offer to check stock.\n"
        "Do not ask for additional search details."
    )
    return Agent(
        name="RetailNext Assistant",
        instructions=instructions,
        model=LLM_MODEL,
        model_settings=ModelSettings()
    )


def run_agent_summary(user_input: str, initial: list, reranked: list, user_profile: dict, event: dict | None, last_context: str) -> dict | str:
    agent = st.session_state.agent
    history = st.session_state.agent_messages[-10:]

    # Build a compact payload to avoid messy formatting
    top3 = [
        {
            "name": p.get("name"),
            "price": p.get("price"),
            "reasons": p.get("reasoning", [])[:2]
        }
        for p in reranked[:3]
    ]
    extras = [p.get("name") for p in initial[3:5] if p.get("name")]

    payload = {
        "user_query": user_input,
        "event": event or {},
        "last_conversation_context": last_context,
        "top3": top3,
        "extras": extras
    }

    instruction = (
        "Return ONLY valid JSON with keys: intro, closing, extras_blurb. "
        "intro/closing/extras_blurb must be single-line strings with NO markdown formatting (no *, _, ★, etc.). "
        "Use plain text only. Keep each string under 100 characters. "
        "Use a warm, conversational tone. The closing should ask about adding to cart. "
        "Do not include bullet lists or product names in these strings."
    )
    history.append({
        "role": "user",
        "content": f"{instruction}\n\n{json.dumps(payload, indent=2)}"
    })
    result = Runner.run_sync(agent, history)
    response_text = result.final_output or ""

    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    st.session_state.agent_messages.append({"role": "assistant", "content": response_text})
    st.session_state.agent_messages = st.session_state.agent_messages[-10:]

    parsed = _extract_json(response_text)
    return parsed if isinstance(parsed, dict) else response_text

# =============================================================================
# UI
# =============================================================================

def render_product_cards(products: list, show_vector_score: bool = False, max_items: int = 5):
    if not products:
        return
    cols = st.columns(min(len(products), max_items))
    for i, product in enumerate(products[:max_items]):
        with cols[i]:
            with st.container():
                image_url = product.get("image_url")
                article_type = product.get("type", "Item")
                color = product.get("color", "")

                icons = {
                    "Dress": "👗",
                    "Shirt": "👔",
                    "Tops": "👚",
                    "Heels": "👠",
                    "Flats": "🥿",
                    "Trousers": "👖",
                    "Jeans": "👖",
                    "Skirts": "👗",
                    "Kurtas": "👘",
                    "Sarees": "🥻"
                }

                if image_url:
                    st.image(image_url, width=200)
                else:
                    st.markdown(f"### {icons.get(article_type, '👕')}")

                name = product.get("name", "Unknown")
                st.markdown(f"**{name[:40]}**" if len(name) > 40 else f"**{name}**")

                price = product.get("price", 0)
                st.markdown(f"💰 **${price:.2f}**")

                if color:
                    st.caption(f"Color: {color}")

                if show_vector_score:
                    vscore = product.get("vector_score", 0)
                    if vscore:
                        st.caption(f"Vector score: {vscore}")


def render_reranked_cards(products: list):
    if not products:
        return
    cols = st.columns(min(len(products), 3))
    for i, product in enumerate(products[:3]):
        with cols[i]:
            with st.container():
                image_url = product.get("image_url")
                article_type = product.get("type", "Item")
                color = product.get("color", "")

                icons = {
                    "Dress": "👗",
                    "Shirt": "👔",
                    "Tops": "👚",
                    "Heels": "👠",
                    "Flats": "🥿",
                    "Trousers": "👖",
                    "Jeans": "👖",
                    "Skirts": "👗",
                    "Kurtas": "👘",
                    "Sarees": "🥻"
                }

                if image_url:
                    st.image(image_url, width=200)
                else:
                    st.markdown(f"### {icons.get(article_type, '👕')}")

                name = product.get("name", "Unknown")
                st.markdown(f"**{name[:40]}**" if len(name) > 40 else f"**{name}**")

                price = product.get("price", 0)
                st.markdown(f"💰 **${price:.2f}**")

                if color:
                    st.caption(f"Color: {color}")

                event_score = product.get("event_score", 0)
                pref_score = product.get("preference_score", 0)
                overall = product.get("overall_score", 0)

                # Show match score as a progress bar
                st.caption("**Match Score:**")
                st.progress(overall, text=f"{overall:.0%}")
                st.caption(f"Event: {event_score:.0%} | Preference: {pref_score:.0%}")

                # Show top 2 reasons in a cleaner format
                reasons = product.get("reasoning", []) or []
                if reasons:
                    st.caption("**Why this works:**")
                    for r in reasons[:2]:
                        # Clean any markdown characters
                        clean_r = r.replace("*", "").replace("_", "").replace("★", "").strip()
                        st.caption(f"✓ {clean_r[:80]}")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Add to cart", key=f"addcart_{product.get('id')}_{i}"):
                        st.session_state.cart.append({
                            "id": product.get("id"),
                            "name": product.get("name"),
                            "price": product.get("price"),
                            "quantity": 1
                        })
                        st.session_state.await_payment = True
                        st.success("Added to cart.")
                with col_b:
                    if st.button("Try on", key=f"tryon_{product.get('id')}_{i}"):
                        st.session_state.tryon_product = product
                        st.session_state.show_tryon_panel = True
                        st.session_state.tryon_result_b64 = None
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Try-on panel opened below. You can upload a photo or paste a URL."
                        })


def stream_paragraphs(paragraphs: list[str], delay: float = 0.03):
    """Yield paragraph chunks for streaming display."""
    for p in paragraphs:
        yield p + "\n\n"
        time.sleep(delay)


def render_message_content(content: str, msg: dict):
    """Render message content, parsing JSON summaries if needed."""
    if not content:
        return
    # Try to detect JSON summary blob from earlier runs
    parsed = None
    if isinstance(content, str) and content.strip().startswith("{"):
        parsed = _extract_json(content)

    if isinstance(parsed, dict) and {"intro", "closing", "extras_blurb"} <= set(parsed.keys()):
        intro = parsed.get("intro", "")
        extras_blurb = parsed.get("extras_blurb", "")
        closing = parsed.get("closing", "")
        if intro:
            st.markdown(intro)
        st.markdown("**Top recommendations:**")
        for item in (msg.get("reranked_products") or [])[:3]:
            reasons = item.get("reasoning", [])[:2]
            reason_text = " ".join(reasons)
            st.markdown(f"- **{item.get('name', 'Item')}** — {reason_text}")
        initial = msg.get("initial_products") or []
        if initial[3:5]:
            extra_names = [p.get("name") for p in initial[3:5] if p.get("name")]
            if extras_blurb:
                st.markdown(extras_blurb)
            st.markdown(f"Extra ideas: {', '.join(extra_names)}")
        if closing:
            st.markdown(closing)
        return

    st.markdown(content)


def render_tryon_panel():
    """Render persistent try-on panel when user clicks Try On."""
    try:
        if not st.session_state.show_tryon_panel or not st.session_state.tryon_product:
            if st.session_state.show_tryon_panel and not st.session_state.tryon_product:
                st.warning("Try-on is enabled but no product was selected.")
            return

        st.markdown("---")
        st.markdown("### 👗 Virtual Try-On")
        st.caption("Provide a photo to generate a try-on preview.")
        st.caption(f"Selected item: {st.session_state.tryon_product.get('name', 'Item')}")

        method = st.radio(
            "Choose a photo source:",
            ["Upload a photo", "Use a photo URL", "Use profile photo"],
            horizontal=True,
            key="tryon_source_main"
        )

        user_image_input = None
        if method == "Upload a photo":
            uploaded = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"],
                key="tryon_upload_main"
            )
            if uploaded:
                user_bytes = uploaded.read()
                mime = uploaded.type or "image/jpeg"
                user_image_input = {
                    "bytes": user_bytes,
                    "mime": mime,
                    "debug": {"source": "upload", "bytes": len(user_bytes), "mime": mime}
                }
        elif method == "Use a photo URL":
            url = st.text_input("Photo URL", key="tryon_url_main")
            if url:
                user_image_input = {"url": url}
        else:
            profile = st.session_state.user_profile or {}
            photo_url = profile.get("photo_url") or profile.get("profile_photo_url")
            if photo_url:
                st.caption(f"Using profile photo: {photo_url}")
                user_image_input = {"url": photo_url}
            else:
                st.warning("No profile photo found.")

        if st.button("Generate Try-On", key="tryon_generate_main"):
            if not user_image_input:
                st.warning("Please provide a photo first.")
            else:
                with st.spinner("Generating virtual try-on..."):
                    product_img = st.session_state.tryon_product.get("image_url")
                    result_b64, debug = generate_tryon_image(user_image_input, product_img)
                    if result_b64:
                        st.session_state.tryon_result_b64 = result_b64
                        st.session_state.tryon_debug = debug
                    else:
                        st.session_state.tryon_debug = debug
                        err_text = (debug or {}).get("error", "") if isinstance(debug, dict) else ""
                        if "verify" in err_text.lower() or "organization" in err_text.lower():
                            st.error(
                                "Try-on is unavailable because the OpenAI organization is not verified for image generation."
                            )
                            st.caption("Please verify the organization in OpenAI settings, then retry.")
                        else:
                            st.error("Try-on generation failed. Please try a different photo.")

        if st.session_state.tryon_result_b64:
            st.image(base64.b64decode(st.session_state.tryon_result_b64), caption="Virtual Try-On")

        if st.session_state.get("tryon_debug"):
            with st.expander("Try-on debug details", expanded=False):
                st.code(st.session_state.tryon_debug)
    except Exception as exc:
        st.error(f"Try-on panel error: {exc}")


def normalize_message_content(msg: dict) -> bool:
    """Normalize any JSON summary blobs into readable paragraphs in session state."""
    content = msg.get("content", "")
    if not isinstance(content, str) or not content.strip().startswith("{"):
        return False
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        return False
    if not {"intro", "closing", "extras_blurb"} <= set(parsed.keys()):
        return False

    intro = parsed.get("intro", "")
    extras_blurb = parsed.get("extras_blurb", "")
    closing = parsed.get("closing", "")
    paragraphs = []
    if intro:
        paragraphs.append(intro)
    paragraphs.append("**Top recommendations:**")
    for item in (msg.get("reranked_products") or [])[:3]:
        reasons = item.get("reasoning", [])[:2]
        reason_text = " ".join(reasons)
        paragraphs.append(f"- **{item.get('name', 'Item')}** — {reason_text}")
    initial = msg.get("initial_products") or []
    if initial[3:5]:
        extra_names = [p.get("name") for p in initial[3:5] if p.get("name")]
        if extras_blurb:
            paragraphs.append(extras_blurb)
        paragraphs.append(f"Extra ideas: {', '.join(extra_names)}")
    if closing:
        paragraphs.append(closing)

    msg["content"] = "\n\n".join(paragraphs)
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title(f"{APP_ICON} {APP_TITLE}")

    if db is None:
        st.error("Database connection failed. Check your MongoDB URI.")
        return

    if openai_client is None:
        st.error("OpenAI client initialization failed. Check your API key.")
        return

    # Session state
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid4().hex[:8]}"
    if "pending" not in st.session_state:
        st.session_state.pending = {
            "active": False,
            "needs_category": False,
            "needs_color": False,
            "category": "",
            "color": "",
            "await_event_confirm": False,
            "saved_query": "",
            "saved_category": "",
            "saved_color": "",
            "saved_gender_override": "",
        }
    if "event_context" not in st.session_state:
        st.session_state.event_context = None
    if "event_context_source" not in st.session_state:
        st.session_state.event_context_source = ""
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent()
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "last_conversation_context" not in st.session_state:
        st.session_state.last_conversation_context = ""
    if "cart" not in st.session_state:
        st.session_state.cart = []
    if "await_add_to_cart" not in st.session_state:
        st.session_state.await_add_to_cart = False
    if "await_payment" not in st.session_state:
        st.session_state.await_payment = False
    if "tryon_product" not in st.session_state:
        st.session_state.tryon_product = None
    if "show_tryon_panel" not in st.session_state:
        st.session_state.show_tryon_panel = False
    if "tryon_result_b64" not in st.session_state:
        st.session_state.tryon_result_b64 = None
    if "tryon_debug" not in st.session_state:
        st.session_state.tryon_debug = None
    if "last_initial_results" not in st.session_state:
        st.session_state.last_initial_results = []
    if "last_reranked_results" not in st.session_state:
        st.session_state.last_reranked_results = []

    # Sidebar: user selection + profile context + recent conversations
    with st.sidebar:
        st.markdown("## 👤 Select User")
        users = get_all_users()
        if not users:
            st.error("No users found")
            return

        user_options = {f"{u.get('name', 'Unknown')} ({u.get('email', '')})": u["_id"] for u in users}
        user_names = list(user_options.keys())

        current_index = 0
        if st.session_state.current_user_id in user_options.values():
            for idx, u in enumerate(users):
                if u["_id"] == st.session_state.current_user_id:
                    current_index = idx
                    break

        selected_name = st.selectbox("Choose a demo user:", user_names, index=current_index)
        selected_user_id = user_options[selected_name]

        if selected_user_id != st.session_state.current_user_id:
            st.session_state.current_user_id = selected_user_id
            st.session_state.user_profile = get_user_profile(selected_user_id)
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.agent_messages = []
            st.session_state.session_id = f"session_{uuid4().hex[:8]}"
            st.session_state.event_context = None
            st.session_state.event_context_source = ""
            st.session_state.pending = {
                "active": False,
                "needs_category": False,
                "needs_color": False,
                "category": "",
                "color": "",
                "await_event_confirm": False,
                "saved_query": "",
                "saved_category": "",
                "saved_color": "",
                "saved_gender_override": "",
            }
            st.session_state.last_conversation_context = get_last_conversation_context(selected_user_id)
            st.session_state.await_add_to_cart = False
            st.session_state.await_payment = False
            st.session_state.tryon_product = None
            st.session_state.show_tryon_panel = False
            st.session_state.tryon_result_b64 = None
            st.session_state.tryon_debug = None
            st.rerun()

        if st.session_state.user_profile:
            profile = st.session_state.user_profile
            st.divider()
            st.markdown("### Profile")
            st.markdown(f"**{profile.get('name', 'Unknown')}**")
            st.caption(profile.get("email", ""))
            prefs = profile.get("preferences", {})
            style_prefs = prefs.get("style_preferences", [])
            colors = prefs.get("favorite_colors", [])
            price_range = prefs.get("price_range", {})
            sizes = profile.get("sizes", {})

            if style_prefs:
                st.caption(f"Style: {', '.join(style_prefs[:4])}")
            if colors:
                st.caption(f"Colors: {', '.join(colors[:4])}")
            if price_range:
                min_p = price_range.get("min", 0)
                max_p = price_range.get("max", 0)
                st.caption(f"Budget: ${min_p} - ${max_p}")
            if sizes:
                st.caption(
                    f"Sizes: Top {sizes.get('top', 'N/A')}, "
                    f"Bottom {sizes.get('bottom', 'N/A')}, "
                    f"Dress {sizes.get('dress', 'N/A')}, "
                    f"Shoes {sizes.get('shoes', 'N/A')}"
                )

            events = profile.get("upcoming_events", [])
            if events:
                event = events[0]
                st.divider()
                st.markdown("### Upcoming Event")
                st.caption(f"Type: {event.get('event_type', 'Event')}")
                if event.get("date"):
                    st.caption(f"Date: {event.get('date')}")
                if event.get("dress_code"):
                    st.caption(f"Dress code: {event.get('dress_code')}")
                if event.get("venue"):
                    st.caption(f"Venue: {event.get('venue')}")
                if event.get("notes"):
                    st.caption(f"Notes: {event.get('notes')}")

        st.divider()
        st.markdown("### Recent Conversations")
        if st.session_state.current_user_id:
            recent = get_user_conversations(st.session_state.current_user_id, limit=3)
            if not recent:
                st.caption("No recent conversations.")
            else:
                for convo in recent:
                    updated_at = convo.get("updated_at", "")
                    messages = convo.get("messages", []) or []
                    snippet = ""
                    if messages:
                        last_msg = messages[-1].get("content", "")
                        snippet = (last_msg[:80] + "...") if len(last_msg) > 80 else last_msg
                    st.markdown(f"**{updated_at}**")
                    if snippet:
                        st.caption(snippet)
                    st.divider()

    # Render history (chat bubbles + previous results)
    for msg in st.session_state.messages:
        normalize_message_content(msg)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message_content(msg.get("content", ""), msg)
            if msg.get("initial_products"):
                with st.expander("Initial 5 results (vector search)", expanded=False):
                    render_product_cards(msg["initial_products"], show_vector_score=True, max_items=5)
            if msg.get("reranked_products"):
                st.markdown("**Top 3 reranked results**")
                render_reranked_cards(msg["reranked_products"])

    # Persistent try-on panel (after history)
    render_tryon_panel()

    # Initial greeting
    if not st.session_state.messages:
        greeting = "Hello! What are you looking for today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)

    if not st.session_state.user_profile:
        st.info("Select a user from the sidebar to start searching.")
        return

    # Chat input and main flow
    if user_input := st.chat_input("Search for products (e.g., red dress)"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        pending = st.session_state.pending
        last_context = st.session_state.get("last_conversation_context", "")

        # Quick actions: stock check / add to cart / checkout
        if not pending["await_event_confirm"]:
            if wants_check_stock(user_input):
                products = st.session_state.last_reranked_results or st.session_state.last_initial_results
                with st.chat_message("assistant"):
                    if not products:
                        msg = "No recent results to check. Please search for products first."
                        st.markdown(msg)
                    else:
                        st.markdown("**Stock availability:**")
                        for i, p in enumerate(products[:3], 1):
                            stock = p.get("stock_quantity")
                            stock_text = f"{stock} available" if stock is not None else "Stock info unavailable"
                            name = p.get('name', 'Item')
                            st.caption(f"{i}. **{name}**: {stock_text}")

                        msg = "Stock checked. Which item would you like to add to your cart? Just tell me the name or number."
                        st.markdown(msg)
                        st.session_state.await_add_to_cart = True

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg if products else "No recent results to check."
                })
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": msg if products else "No recent results to check.",
                    "timestamp": datetime.now().isoformat()
                })
                save_conversation(
                    st.session_state.current_user_id,
                    st.session_state.session_id,
                    st.session_state.conversation_history
                )
                return

            if st.session_state.await_add_to_cart:
                products = st.session_state.last_reranked_results or st.session_state.last_initial_results
                if not products:
                    msg = "I don't have any recent results. Please search for products first."
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                    st.session_state.await_add_to_cart = False
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    return

                # Try to find product by name/brand in user input
                selected_item = None
                user_lower = user_input.lower()

                # Check if user specified a number (1, 2, 3)
                for i in range(1, min(len(products) + 1, 10)):
                    if str(i) in user_input or f"#{i}" in user_input:
                        selected_item = products[i - 1]
                        break

                # If no number, try to match product name/brand
                if not selected_item:
                    for p in products[:3]:
                        name_lower = p.get("name", "").lower()
                        # Check if any significant word from product name is in user input
                        name_words = [w for w in name_lower.split() if len(w) > 3]
                        if any(word in user_lower for word in name_words):
                            selected_item = p
                            break

                # If still not found, default to first item if user said "add" or "yes"
                if not selected_item and wants_add_to_cart(user_input):
                    selected_item = products[0]

                if selected_item:
                    st.session_state.cart.append({
                        "id": selected_item.get("id"),
                        "name": selected_item.get("name"),
                        "price": selected_item.get("price"),
                        "quantity": 1
                    })
                    msg = f"✅ Added **{selected_item.get('name', 'item')}** to your cart (${selected_item.get('price', 0):.2f}). Would you like to proceed to payment?"
                    st.session_state.await_payment = True
                    st.session_state.await_add_to_cart = False

                    with st.chat_message("assistant"):
                        st.markdown(msg)

                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_conversation(
                        st.session_state.current_user_id,
                        st.session_state.session_id,
                        st.session_state.conversation_history
                    )
                else:
                    msg = "I couldn't find that item in the recent results. Could you tell me which one by name or number (1, 2, 3)?"
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

                return

            if st.session_state.await_payment:
                answer = user_input.strip().lower()
                is_yes = answer in ["yes", "y", "sure", "ok", "okay", "proceed", "pay", "checkout"]
                is_no = answer in ["no", "n", "nope", "not yet", "cancel"]

                if is_yes:
                    cart = st.session_state.get("cart", [])
                    total = sum(item["price"] * item.get("quantity", 1) for item in cart)

                    with st.chat_message("assistant"):
                        st.markdown(f"**Order Summary:**")
                        for item in cart:
                            st.caption(f"• {item['name']}: ${item['price']:.2f}")
                        st.markdown(f"**Total: ${total:.2f}**")
                        st.markdown("---")
                        st.markdown("✅ **Payment successful!** Your order is confirmed. Thank you for shopping!")
                        st.caption("🚚 Delivery is planned. You’ll receive a confirmation email shortly.")

                    msg = f"Payment successful. Order total: ${total:.2f}. Delivery is planned."
                    st.session_state.await_payment = False
                    st.session_state.cart = []  # Clear cart after payment

                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_conversation(
                        st.session_state.current_user_id,
                        st.session_state.session_id,
                        st.session_state.conversation_history
                    )
                    return
                elif is_no:
                    msg = "No problem! Your items are still in the cart. Let me know when you're ready to checkout or if you'd like to search for more items."
                    st.session_state.await_payment = False

                    with st.chat_message("assistant"):
                        st.markdown(msg)

                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_conversation(
                        st.session_state.current_user_id,
                        st.session_state.session_id,
                        st.session_state.conversation_history
                    )
                    return

            if wants_checkout(user_input):
                cart = st.session_state.get("cart", [])
                if not cart:
                    msg = "Your cart is empty. Please add some items first!"
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    return

                total = sum(item["price"] * item.get("quantity", 1) for item in cart)
                with st.chat_message("assistant"):
                    st.markdown(f"**Order Summary:**")
                    for item in cart:
                        st.caption(f"• {item['name']}: ${item['price']:.2f}")
                    st.markdown(f"**Total: ${total:.2f}**")
                    st.markdown("---")
                    st.markdown("Would you like to proceed with payment?")

                st.session_state.await_payment = True
                msg = f"Checkout initiated. Total: ${total:.2f}. Proceed with payment?"

                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": msg,
                    "timestamp": datetime.now().isoformat()
                })
                save_conversation(
                    st.session_state.current_user_id,
                    st.session_state.session_id,
                    st.session_state.conversation_history
                )
                return
        # If we are waiting on missing fields
        if pending["active"]:
            intent = infer_intent_llm(user_input, last_context)
            inferred_category = intent.get("category", "") if isinstance(intent, dict) else ""
            inferred_color = intent.get("color", "") if isinstance(intent, dict) else ""
            if pending["needs_category"]:
                pending["category"] = inferred_category or extract_category(user_input) or pending["category"]
                pending["needs_category"] = not bool(pending["category"])
            if pending["needs_color"]:
                pending["color"] = inferred_color or extract_color(user_input) or pending["color"]
                pending["needs_color"] = not bool(pending["color"])

            if pending["needs_category"] or pending["needs_color"]:
                question = ""
                if pending["needs_category"] and pending["needs_color"]:
                    question = "For the wedding, what category (dress, t-shirt, shoes, etc.) and what color are you looking for?"
                elif pending["needs_category"]:
                    question = "What category are you shopping for (dress, t-shirt, shoes, etc.)?"
                else:
                    question = "What color are you looking for?"

                with st.chat_message("assistant"):
                    st.markdown(question)

                st.session_state.messages.append({"role": "assistant", "content": question})
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": question,
                    "timestamp": datetime.now().isoformat()
                })
                save_conversation(
                    st.session_state.current_user_id,
                    st.session_state.session_id,
                    st.session_state.conversation_history
                )
                return

            # We have required fields now
            pending["active"] = False

        # If awaiting event confirmation, handle yes/no (no repeated nagging)
        if pending["await_event_confirm"]:
            answer = user_input.strip().lower()
            is_yes = answer in ["yes", "y", "sure", "ok", "okay"]
            is_no = answer in ["no", "n", "nope", "not now"]

            if is_yes or is_no:
                if is_yes:
                    profile = st.session_state.user_profile or {}
                    events = profile.get("upcoming_events", [])
                    if events:
                        st.session_state.event_context = events[0]
                        st.session_state.event_context_source = "profile"
                else:
                    st.session_state.event_context = None
                    st.session_state.event_context_source = "declined"

                # Resume saved query
                user_input = pending["saved_query"]
                category = pending["saved_category"]
                color = pending["saved_color"]
                gender_override = pending["saved_gender_override"] or None

                pending["await_event_confirm"] = False
                pending["saved_query"] = ""
                pending["saved_category"] = ""
                pending["saved_color"] = ""
                pending["saved_gender_override"] = ""
            else:
                # If unclear, proceed without using the upcoming event (no repeat prompt)
                st.session_state.event_context = None
                st.session_state.event_context_source = "declined"
                pending["await_event_confirm"] = False
                pending["saved_query"] = ""
                pending["saved_category"] = ""
                pending["saved_color"] = ""
                pending["saved_gender_override"] = ""

        # New intent detection
        intent = infer_intent_llm(user_input, last_context)
        category = intent.get("category", "") if isinstance(intent, dict) else ""
        color = intent.get("color", "") if isinstance(intent, dict) else ""
        gender_override = intent.get("gender_override", "") if isinstance(intent, dict) else ""
        event_type = intent.get("event_type", "") if isinstance(intent, dict) else ""

        # Fallback to lightweight rules if model returns nothing
        if not category:
            category = extract_category(user_input)
        if not color:
            color = extract_color(user_input)
        if not gender_override:
            gender_override = detect_gender_override(user_input) or ""
        if not event_type:
            event_type = extract_event_type(user_input)

        wedding_intent = event_type == "wedding" or detect_wedding_intent(user_input)

        # Store explicit event context from user input
        if event_type:
            st.session_state.event_context = {
                "event_type": event_type,
                "date": "",
                "dress_code": "",
                "venue": "",
                "notes": "User-stated event"
            }
            st.session_state.event_context_source = "user"

        if wedding_intent and (not category or not color):
            pending["active"] = True
            pending["category"] = category
            pending["color"] = color
            pending["needs_category"] = not bool(category)
            pending["needs_color"] = not bool(color)

            question = ""
            if pending["needs_category"] and pending["needs_color"]:
                question = "For the wedding, what category (dress, t-shirt, shoes, etc.) and what color are you looking for?"
            elif pending["needs_category"]:
                question = "What category are you shopping for (dress, t-shirt, shoes, etc.)?"
            else:
                question = "What color are you looking for?"

            with st.chat_message("assistant"):
                st.markdown(question)

            st.session_state.messages.append({"role": "assistant", "content": question})
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            save_conversation(
                st.session_state.current_user_id,
                st.session_state.session_id,
                st.session_state.conversation_history
            )
            return

        # If color is missing (non-wedding flow), ask for it
        if not wedding_intent and not color:
            pending["active"] = True
            pending["category"] = category
            pending["color"] = ""
            pending["needs_category"] = False
            pending["needs_color"] = True

            question = "What color are you looking for?"
            with st.chat_message("assistant"):
                st.markdown(question)

            st.session_state.messages.append({"role": "assistant", "content": question})
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            save_conversation(
                st.session_state.current_user_id,
                st.session_state.session_id,
                st.session_state.conversation_history
            )
            return

        # If no event mentioned and no stored event, ask to use upcoming event
        if not event_type and not st.session_state.event_context:
            profile = st.session_state.user_profile or {}
            events = profile.get("upcoming_events", [])
            if events:
                event = events[0]
                event_summary = f"{event.get('event_type', 'event')}"
                if event.get("date"):
                    event_summary += f" on {event.get('date')}"

                question = (
                    f"I noticed your upcoming {event_summary}. "
                    f"If you'd like me to use it to rank results, reply 'yes'. "
                    f"Otherwise, just continue and I won't use it."
                )

                pending["await_event_confirm"] = True
                pending["saved_query"] = user_input
                pending["saved_category"] = category
                pending["saved_color"] = color
                pending["saved_gender_override"] = gender_override or ""

                with st.chat_message("assistant"):
                    st.markdown(question)

                st.session_state.messages.append({"role": "assistant", "content": question})
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": question,
                    "timestamp": datetime.now().isoformat()
                })
                save_conversation(
                    st.session_state.current_user_id,
                    st.session_state.session_id,
                    st.session_state.conversation_history
                )
                return
        # Proceed with search + rerank via agent
        with st.chat_message("assistant"):
            steps_container = st.container()
            steps_container.markdown("**Step 1/3:** Running vector search (top 5)...")

            # Vector search (only category + color)
            user_profile = st.session_state.user_profile or {}
            event = st.session_state.event_context

            search_query = build_search_query(category, color, user_input)
            gender_filter = get_gender_filter(user_profile, gender_override or None)

            embedding = generate_embedding(search_query)
            results = vector_search_products(embedding, limit=5, gender_filter=gender_filter) if embedding else []
            initial = format_products_for_ui(results)

            steps_container.markdown(f"✅ **Step 1/3 complete:** Found {len(initial)} results.")

            if initial:
                with st.expander("Initial 5 results (vector search)", expanded=False):
                    render_product_cards(initial, show_vector_score=True, max_items=5)
                if len(initial) < 5:
                    st.caption(f"Only {len(initial)} results available after filtering.")

            # Rerank step
            time.sleep(0.2)
            steps_container.markdown("**Step 2/3:** Reranking with LLM + images (event-first)...")
            reranked = rerank_products_multimodal(initial, user_profile, event, user_input)
            if not reranked:
                reranked = []
                for p in initial[:3]:
                    reranked.append({
                        **p,
                        "event_score": 0.5,
                        "preference_score": 0.5,
                        "overall_score": 0.5,
                        "reasoning": ["Rerank unavailable; using vector results."]
                    })

            steps_container.markdown("✅ **Step 2/3 complete:** Rerank finished.")
            if reranked:
                st.markdown("**🎯 Match Analysis**")
                with st.expander("View detailed reasoning", expanded=False):
                    for item in reranked:
                        st.markdown(f"**{item.get('name', 'Item')}**")
                        for r in item.get("reasoning", [])[:2]:
                            # Clean markdown
                            clean_r = r.replace("*", "").replace("_", "").replace("★", "").strip()
                            st.caption(f"• {clean_r}")

            if reranked:
                st.markdown("**Top 3 reranked results**")
                render_reranked_cards(reranked)

            # Summary step
            time.sleep(0.2)
            steps_container.markdown("**Step 3/3:** Generating summary response...")
            last_context = st.session_state.get("last_conversation_context", "")
            summary_data = run_agent_summary(user_input, initial, reranked, user_profile, event, last_context)
            paragraphs = []

            if isinstance(summary_data, dict):
                intro = summary_data.get("intro", "Here are some great options for you.")
                extras_blurb = summary_data.get("extras_blurb", "")
                closing = summary_data.get(
                    "closing",
                    "Would you like me to add any of these to your cart?"
                )

                # Clean any stray markdown from agent response
                intro = intro.replace("*", "").replace("_", "").replace("★", "").strip()
                closing = closing.replace("*", "").replace("_", "").replace("★", "").strip()
                extras_blurb = extras_blurb.replace("*", "").replace("_", "").replace("★", "").strip()

                if intro:
                    paragraphs.append(intro)

                # Show brief product list
                paragraphs.append("**My top picks for you:**")
                for i, item in enumerate(reranked[:3], 1):
                    name = item.get('name', 'Item')
                    price = item.get('price', 0)
                    # Take only the first reason to keep it concise
                    reason = item.get("reasoning", [""])[0]
                    if reason:
                        # Clean markdown from reason too
                        reason = reason.replace("*", "").replace("_", "").replace("★", "").strip()
                        paragraphs.append(f"{i}. **{name}** (${price:.2f}) — {reason[:100]}")
                    else:
                        paragraphs.append(f"{i}. **{name}** (${price:.2f})")

                if initial[3:5]:
                    extra_names = [p.get("name") for p in initial[3:5] if p.get("name")]
                    if extra_names:
                        if extras_blurb:
                            paragraphs.append(extras_blurb)
                        paragraphs.append(f"*Also worth considering:* {', '.join(extra_names)}")

                if closing:
                    paragraphs.append(closing)
            else:
                # Fallback to simple text
                clean_text = str(summary_data or "Here are some recommendations for you.")
                clean_text = clean_text.replace("*", "").replace("_", "").replace("★", "").strip()
                paragraphs.append(clean_text)

            st.write_stream(stream_paragraphs(paragraphs))
            response = "\n\n".join(paragraphs)

            # Save last results for quick actions
            st.session_state.last_initial_results = initial
            st.session_state.last_reranked_results = reranked
            st.session_state.await_add_to_cart = bool(reranked)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "initial_products": initial,
            "reranked_products": reranked
        })
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        save_conversation(
            st.session_state.current_user_id,
            st.session_state.session_id,
            st.session_state.conversation_history
        )


if __name__ == "__main__":
    main()
