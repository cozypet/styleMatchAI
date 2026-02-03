"""
RetailNext Intelligent Search - Minimal Streamlit Demo
Raw vector search + chat UI results only.
"""

import os
from datetime import datetime

import certifi
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "retail_demo"
COLLECTION_NAME = "productAndEmbeddings"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"

APP_TITLE = "RetailNext Intelligent Search"
APP_ICON = "🛍️"

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
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


def vector_search_products(query_embedding: list, limit: int = 3) -> list:
    if db is None:
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 1,
                "id": 1,
                "productDisplayName": 1,
                "price": 1,
                "baseColour": 1,
                "articleType": 1,
                "image": 1,
                "image_url": 1,
                "imageUrl": 1,
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
            "color": product.get("baseColour", ""),
            "type": product.get("articleType", ""),
            "score": round(product.get("score", 0), 3),
            "image_url": image_url
        })
    return formatted

# =============================================================================
# UI
# =============================================================================

def render_product_cards(products: list):
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
                    st.image(image_url, use_column_width=True)
                else:
                    st.markdown(f"### {icons.get(article_type, '👕')}")

                name = product.get("name", "Unknown")
                st.markdown(f"**{name[:40]}**" if len(name) > 40 else f"**{name}**")

                price = product.get("price", 0)
                st.markdown(f"💰 **${price:.2f}**")

                if color:
                    st.caption(f"Color: {color}")

                score = product.get("score", 0)
                if score > 0:
                    st.progress(score, text=f"Match: {score * 100:.0f}%")


def main():
    st.title(f"{APP_ICON} {APP_TITLE}")

    if db is None:
        st.error("Database connection failed. Check your MongoDB URI.")
        return

    if openai_client is None:
        st.error("OpenAI client initialization failed. Check your API key.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("products"):
                render_product_cards(msg["products"])

    if user_input := st.chat_input("Search for products (e.g., red dress)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                embedding = generate_embedding(user_input)
                results = vector_search_products(embedding, limit=3) if embedding else []
                products = format_products_for_ui(results)

                if products:
                    st.markdown("Here are the top matches:")
                    render_product_cards(products)
                else:
                    st.markdown("No products found.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": "Here are the top matches:" if products else "No products found.",
            "products": products
        })


if __name__ == "__main__":
    main()
