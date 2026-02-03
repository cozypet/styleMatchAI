"""
RetailNext Intelligent Search - Streamlit Demo Application
AI-powered retail shopping assistant with vector search and personalized recommendations.
"""

import os
import json
from datetime import datetime

import streamlit as st
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "retail_demo"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"

# App Configuration
APP_TITLE = "RetailNext Intelligent Search"
APP_ICON = "🛍️"

# =============================================================================
# STREAMLIT PAGE CONFIGURATION (must be first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_mongodb_client():
    """Create and cache MongoDB client connection."""
    if not MONGODB_URI:
        st.error("MONGODB_URI not found in environment variables")
        return None
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    return client

@st.cache_resource
def get_openai_client():
    """Create and cache OpenAI client."""
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment variables")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

# Initialize clients
mongo_client = get_mongodb_client()
openai_client = get_openai_client()

# Get database reference
if mongo_client:
    db = mongo_client[DATABASE_NAME]
else:
    db = None

# =============================================================================
# INTENT MAPPING FOR QUERY TRANSFORMATION
# =============================================================================

# Maps user language to product catalog vocabulary
INTENT_MAP = {
    # Occasion → Usage/Style
    "wedding": "formal printed",
    "interview": "formal solid",
    "date night": "casual",
    "date": "casual",
    "party": "evening printed",
    "office": "formal",
    "vacation": "casual",
    "beach": "casual",
    "garden": "printed",

    # Style → Pattern descriptors
    "elegant": "solid formal",
    "sophisticated": "formal printed",
    "bohemian": "printed",
    "boho": "printed",
    "minimalist": "solid",
    "chic": "printed",
    "classic": "solid formal",
    "professional": "formal solid",
    "casual": "casual",
    "sporty": "sports",
}

# Maps user article keywords to catalog article types (use exact database values)
ARTICLE_KEYWORDS = {
    "dress": "Dresses",
    "dresses": "Dresses",
    "gown": "Dresses",
    "shirt": "Shirts",
    "shirts": "Shirts",
    "top": "Tops",
    "tops": "Tops",
    "blouse": "Tops",
    "tshirt": "Tshirts",
    "t-shirt": "Tshirts",
    "shoes": "Heels",
    "heels": "Heels",
    "flats": "Flats",
    "sandals": "Sandals",
    "pants": "Trousers",
    "trousers": "Trousers",
    "jeans": "Jeans",
    "skirt": "Skirts",
    "kurta": "Kurtas",
    "saree": "Sarees",
    "sari": "Sarees",
}

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_all_users():
    """Fetch all users for the dropdown selector."""
    if db is None:
        return []
    users = list(db["user_profiles"].find({}, {"_id": 1, "name": 1, "email": 1}))
    return users

def get_user_profile(user_id: str) -> dict:
    """Fetch a user profile by user_id."""
    if db is None:
        return None
    user = db["user_profiles"].find_one({"_id": user_id})
    return user

def get_user_conversations(user_id: str, limit: int = 5) -> list:
    """Fetch recent conversations for a user."""
    if db is None:
        return []
    conversations = list(
        db["user_conversation_memory"]
        .find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )
    return conversations

def save_conversation(user_id: str, session_id: str, messages: list):
    """Save conversation to MongoDB."""
    if db is None:
        return None

    now = datetime.now().isoformat()
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages,
        "created_at": messages[0]["timestamp"] if messages else now,
        "updated_at": messages[-1]["timestamp"] if messages else now,
    }

    # Upsert: update if exists, insert if not
    result = db["user_conversation_memory"].update_one(
        {"session_id": session_id},
        {"$set": doc},
        upsert=True
    )
    return result

def generate_embedding(text: str) -> list:
    """Generate embedding using OpenAI text-embedding-3-large."""
    if openai_client is None:
        return []

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def vector_search_products(query_embedding: list, filters: dict = None, limit: int = 10) -> list:
    """
    Execute MongoDB Atlas vector search on productAndEmbeddings collection.

    Args:
        query_embedding: The embedding vector for the search query
        filters: Optional filters (gender, price range, stock)
        limit: Maximum number of results to return

    Returns:
        List of matching products with similarity scores
    """
    if db is None:
        return []

    # Build the vector search pipeline
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
                "sku": 1,
                "stock_quantity": 1,
                "is_new_arrival": 1,
                "popularity_score": 1,
                "baseColour": 1,
                "articleType": 1,
                "gender": 1,
                "masterCategory": 1,
                "subCategory": 1,
                "usage": 1,
                "season": 1,
                "image": 1,
                "image_url": 1,
                "imageUrl": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    # Add filter stage if filters provided
    if filters:
        match_stage = {"$match": {}}

        if "gender" in filters:
            match_stage["$match"]["gender"] = filters["gender"]

        if "price" in filters:
            match_stage["$match"]["price"] = filters["price"]

        if "stock_quantity" in filters:
            match_stage["$match"]["stock_quantity"] = filters["stock_quantity"]

        if "articleType" in filters:
            match_stage["$match"]["articleType"] = filters["articleType"]

        if match_stage["$match"]:
            pipeline.insert(1, match_stage)

    # Execute the aggregation
    try:
        results = list(db["productAndEmbeddings"].aggregate(pipeline))
    except Exception as exc:
        print(f"Vector search error: {exc}")
        return []
    return results


def format_products_for_ui(results: list) -> list:
    """Format product docs into UI-friendly dicts."""
    formatted = []
    for product in results[:3]:
        image_url = (
            product.get("image_url")
            or product.get("imageUrl")
            or product.get("image")
        )
        formatted.append({
            "id": str(product.get("_id", "")),
            "name": product.get("productDisplayName", "Unknown"),
            "price": _coerce_float(product.get("price", 0)),
            "color": product.get("baseColour", ""),
            "type": product.get("articleType", ""),
            "stock": _coerce_int(product.get("stock_quantity", 0)),
            "is_new": product.get("is_new_arrival", False),
            "score": round(product.get("score", 0), 3),
            "image_url": image_url
        })
    return formatted

def get_product_by_id(product_id: str) -> dict:
    """Fetch a single product by its _id."""
    if db is None:
        return None
    from bson import ObjectId
    try:
        product = db["productAndEmbeddings"].find_one({"_id": ObjectId(product_id)})
    except Exception:
        product = db["productAndEmbeddings"].find_one({"_id": product_id})
    return product

def get_complementary_products(product: dict, limit: int = 3) -> list:
    """Find complementary products (accessories) based on a main product."""
    if db is None or product is None:
        return []

    # Define complementary categories
    complementary_map = {
        "Dress": ["Heels", "Flats", "Clutches", "Earrings"],
        "Dresses": ["Heels", "Flats", "Clutches", "Earrings"],
        "Shirts": ["Trousers", "Formal Shoes", "Ties"],
        "Shirt": ["Trousers", "Formal Shoes", "Ties"],
        "Tops": ["Jeans", "Skirts", "Heels"],
        "Kurtas": ["Dupatta", "Flats", "Earrings"],
    }

    article_type = product.get("articleType", "")
    gender = product.get("gender", "")

    # Get complementary article types
    comp_types = complementary_map.get(article_type, ["Heels", "Flats"])

    # Find products that match
    query = {
        "articleType": {"$in": comp_types},
        "gender": {"$in": [gender, "Unisex"]},
        "stock_quantity": {"$gt": 0}
    }

    results = list(db["productAndEmbeddings"].find(query).limit(limit))
    return results

# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _coerce_float(value, default: float = 0.0) -> float:
    """Best-effort float coercion for numeric values (e.g., Decimal128, str)."""
    try:
        from bson.decimal128 import Decimal128
        if isinstance(value, Decimal128):
            return float(value.to_decimal())
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: int = 0) -> int:
    """Best-effort int coercion for numeric values."""
    try:
        from bson.decimal128 import Decimal128
        if isinstance(value, Decimal128):
            return int(value.to_decimal())
        return int(value)
    except (TypeError, ValueError):
        return default


def get_style_preferences(prefs: dict) -> list:
    """Support both legacy 'style' and newer 'style_preferences' keys."""
    if not isinstance(prefs, dict):
        return []
    return prefs.get("style_preferences") or prefs.get("style") or []


def get_size_value(sizes: dict, keys: list[str], default: str = "N/A") -> str:
    """Return the first matching size value from a list of keys."""
    if not isinstance(sizes, dict):
        return default
    for key in keys:
        if key in sizes:
            return sizes.get(key) or default
    return default


def _parse_tool_output_to_products(output) -> list:
    """Parse tool outputs (JSON or list) into a product list."""
    if output is None:
        return []

    data = output
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

    # Ignore cart payloads
    if isinstance(data, dict):
        if "items" in data and "total" in data:
            return []
        if isinstance(data.get("products"), list):
            data = data["products"]
        else:
            return []

    if not isinstance(data, list):
        return []

    items = [item for item in data if isinstance(item, dict)]
    if not items:
        return []

    # Basic shape check
    has_name = any("name" in item or "productDisplayName" in item for item in items)
    has_price = any("price" in item for item in items)
    if not (has_name and has_price):
        return []

    return items


def extract_products_from_run(result) -> list:
    """Extract product lists from Agents SDK tool outputs."""
    if not result:
        return []

    products = []
    for item in getattr(result, "new_items", []):
        if getattr(item, "type", "") != "tool_call_output_item":
            continue
        parsed = _parse_tool_output_to_products(getattr(item, "output", None))
        if parsed:
            products = parsed  # Keep the latest tool output

    return products

# =============================================================================
# QUERY TRANSFORMATION FOR VECTOR SEARCH
# =============================================================================

def transform_query_for_search(user_query: str, user_profile: dict, event: dict = None) -> str:
    """
    Transform natural language query into product-catalog-friendly search text.

    Maps user language (elegant, wedding, etc.) to catalog vocabulary (formal, printed, etc.)
    to improve vector search embedding matches.

    Args:
        user_query: The user's natural language query
        user_profile: User profile dict with preferences
        event: Optional upcoming event dict

    Returns:
        Transformed search string matching productDisplayName patterns
    """
    query_lower = user_query.lower()
    # Use a set to track unique terms (lowercase for comparison)
    seen_terms = set()
    catalog_terms = []

    def add_term(term):
        """Add term if not already present (case-insensitive dedup)."""
        term_lower = term.lower()
        if term_lower not in seen_terms:
            seen_terms.add(term_lower)
            catalog_terms.append(term)

    def add_terms(terms_str):
        """Add multiple space-separated terms."""
        for term in terms_str.split():
            add_term(term)

    # Get user's gender preference
    gender = user_profile.get("gender_preference", "Women")
    add_term(gender)

    # Map user intent to catalog vocabulary using INTENT_MAP
    for user_word, catalog_word in INTENT_MAP.items():
        if user_word in query_lower:
            add_terms(catalog_word)

    # Extract article type from query
    for keyword, article in ARTICLE_KEYWORDS.items():
        if keyword in query_lower:
            add_term(article)
            break

    # Get user's color preferences
    colors = user_profile.get("preferences", {}).get("favorite_colors", [])
    if colors:
        # Add top color preference
        add_term(colors[0])

    # Add event context if available
    if event:
        dress_code = event.get("dress_code", "").lower()
        venue = event.get("venue", "").lower()

        # Map dress code to catalog terms
        if dress_code in ["semi-formal", "formal", "black tie"]:
            add_term("formal")

        # Map venue to style suggestions
        if "garden" in venue or "outdoor" in venue:
            add_term("printed")
        elif "beach" in venue:
            add_term("casual")

    return " ".join(catalog_terms)


def search_products_with_context(user_message: str, user_profile: dict, limit: int = 10) -> list:
    """
    Full search pipeline: transform query, generate embedding, execute vector search.

    Args:
        user_message: User's natural language query
        user_profile: User profile with preferences
        limit: Maximum number of results

    Returns:
        List of matching products with similarity scores
    """
    # Initialize debug log in session state
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []

    debug_info = {"timestamp": datetime.now().isoformat()}

    # Get upcoming event if available
    upcoming_events = user_profile.get("upcoming_events", [])
    event = upcoming_events[0] if upcoming_events else None

    # Transform query to catalog vocabulary
    search_query = transform_query_for_search(user_message, user_profile, event)
    debug_info["original_query"] = user_message
    debug_info["transformed_query"] = search_query

    # Generate embedding from transformed query
    query_embedding = generate_embedding(search_query)
    debug_info["embedding_generated"] = bool(query_embedding)
    debug_info["embedding_dimensions"] = len(query_embedding) if query_embedding else 0

    if not query_embedding:
        debug_info["error"] = "Failed to generate embedding"
        st.session_state.debug_log.append(debug_info)
        return []

    # Build filters from user profile
    filters = {}

    # Gender filter
#    gender = user_profile.get("gender_preference", "Women")
    # Map user gender preference to product gender field
#    gender_map = {"Women": "Women", "Men": "Men", "Unisex": "Unisex"}
#    product_gender = gender_map.get(gender, gender)
#    filters["gender"] = {"$in": [product_gender, "Unisex"]}

    # Stock filter - only show in-stock items
#    filters["stock_quantity"] = {"$gt": 0}

    # Price filter based on user budget
#    price_range = user_profile.get("preferences", {}).get("price_range", {})
#    if price_range:
#        min_price = price_range.get("min", 0)
#        max_price = price_range.get("max", 9999)
#        filters["price"] = {"$gte": min_price, "$lte": max_price}

    debug_info["filters"] = str(filters)

    # Execute vector search
    results = vector_search_products(query_embedding, filters, limit)

    # Fallbacks if no results
    fallback_attempts = []
    if not results:
        # Try removing price filter
        relaxed_filters = dict(filters)
        relaxed_filters.pop("price", None)
        results = vector_search_products(query_embedding, relaxed_filters, limit)
        fallback_attempts.append({"removed": "price", "count": len(results)})

    if not results:
        # Try removing gender filter too
        relaxed_filters = dict(filters)
        relaxed_filters.pop("price", None)
        relaxed_filters.pop("gender", None)
        results = vector_search_products(query_embedding, relaxed_filters, limit)
        fallback_attempts.append({"removed": "price+gender", "count": len(results)})

    if not results:
        # Try embedding the raw user query
        raw_embedding = generate_embedding(user_message)
        if raw_embedding:
            results = vector_search_products(raw_embedding, filters, limit)
            fallback_attempts.append({"raw_query": True, "count": len(results)})

    if fallback_attempts:
        debug_info["fallback_attempts"] = fallback_attempts

    debug_info["results_count"] = len(results)
    debug_info["results_preview"] = [
        {"name": r.get("productDisplayName", "N/A")[:50], "score": r.get("score", 0)}
        for r in results[:5]
    ]

    # Log to console for debugging
    print(f"\n=== SEARCH DEBUG ===")
    print(f"Original Query: {user_message}")
    print(f"Transformed Query: {search_query}")
    print(f"Filters: {filters}")
    print(f"Results Count: {len(results)}")
    for r in results[:3]:
        print(f"  - {r.get('productDisplayName', 'N/A')} (score: {r.get('score', 0):.4f})")
    print(f"====================\n")

    st.session_state.debug_log.append(debug_info)

    return results


# =============================================================================
# MATCH SCORING ALGORITHM
# =============================================================================

def calculate_match_score(product: dict, user_profile: dict, event: dict = None) -> dict:
    """
    Calculate match scores for a product based on user profile and event.

    Scoring breakdown:
    - Event Fit (25%): How well product matches event requirements
    - Style Match (25%): Alignment with user's style preferences
    - Availability (25%): Stock availability scoring
    - Trending (25%): Product popularity score

    Args:
        product: Product document from database
        user_profile: User profile with preferences
        event: Optional upcoming event

    Returns:
        Dict with individual scores and overall weighted score
    """
    scores = {
        "event_fit": 0.0,
        "style_match": 0.0,
        "availability": 0.0,
        "trending": 0.0,
        "overall": 0.0
    }

    # Weight configuration
    weights = {
        "event_fit": 0.25,
        "style_match": 0.25,
        "availability": 0.25,
        "trending": 0.25
    }

    # --- Event Fit Score (25%) ---
    if event:
        event_score = 0.0
        dress_code = event.get("dress_code", "").lower()
        venue = event.get("venue", "").lower()
        product_usage = product.get("usage", "").lower()

        # Dress code matching
        if dress_code in ["formal", "black tie"]:
            if product_usage == "formal":
                event_score += 50
            elif product_usage == "casual":
                event_score += 20
        elif dress_code in ["semi-formal", "cocktail"]:
            if product_usage in ["formal", "casual"]:
                event_score += 40
        elif dress_code == "casual":
            if product_usage == "casual":
                event_score += 50

        # Venue appropriateness
        if "garden" in venue or "outdoor" in venue:
            # Printed patterns work well for outdoor events
            if "printed" in product.get("productDisplayName", "").lower():
                event_score += 30
            else:
                event_score += 15
        elif "beach" in venue:
            if product_usage == "casual":
                event_score += 30
        else:
            event_score += 20  # Neutral venue bonus

        scores["event_fit"] = min(event_score, 100) / 100
    else:
        # No event - give neutral score
        scores["event_fit"] = 0.5

    # --- Style Match Score (25%) ---
    style_score = 0.0
    prefs = user_profile.get("preferences", {})

    # Color matching
    favorite_colors = [c.lower() for c in prefs.get("favorite_colors", [])]
    product_color = product.get("baseColour", "").lower()

    if product_color in favorite_colors:
        # Exact color match
        color_rank = favorite_colors.index(product_color)
        style_score += 50 - (color_rank * 10)  # Higher score for preferred colors
    elif product_color:
        # Partial match for complementary colors
        style_score += 15

    # Style preference matching
    style_prefs = [s.lower() for s in get_style_preferences(prefs)]
    product_name = product.get("productDisplayName", "").lower()

    # Map style preferences to product attributes
    style_keywords = {
        "elegant": ["formal", "solid"],
        "classic": ["formal", "solid", "plain"],
        "bohemian": ["printed", "pattern"],
        "boho": ["printed", "pattern"],
        "minimalist": ["solid", "plain"],
        "trendy": ["printed", "new"],
        "casual": ["casual"],
        "professional": ["formal", "solid"],
        "chic": ["printed", "formal"]
    }

    for pref in style_prefs:
        if pref in style_keywords:
            for keyword in style_keywords[pref]:
                if keyword in product_name:
                    style_score += 15
                    break

    scores["style_match"] = min(style_score, 100) / 100

    # --- Availability Score (25%) ---
    stock = product.get("stock_quantity", 0)

    if stock >= 50:
        scores["availability"] = 1.0
    elif stock >= 20:
        scores["availability"] = 0.8
    elif stock >= 10:
        scores["availability"] = 0.6
    elif stock >= 5:
        scores["availability"] = 0.4
    elif stock > 0:
        scores["availability"] = 0.2
    else:
        scores["availability"] = 0.0

    # Bonus for new arrivals
    if product.get("is_new_arrival", False):
        scores["availability"] = min(scores["availability"] + 0.1, 1.0)

    # --- Trending Score (25%) ---
    popularity = product.get("popularity_score", 0)

    # Normalize popularity (assuming 0-100 scale)
    scores["trending"] = min(popularity / 100, 1.0)

    # --- Calculate Overall Score ---
    scores["overall"] = sum(
        scores[key] * weights[key]
        for key in weights.keys()
    )

    # Round all scores to 2 decimal places
    for key in scores:
        scores[key] = round(scores[key], 2)

    return scores


def get_match_explanation(product: dict, scores: dict, user_profile: dict, event: dict = None) -> str:
    """
    Generate a human-readable explanation of why a product matches.

    Args:
        product: The product document
        scores: Match scores from calculate_match_score
        user_profile: User profile
        event: Optional event

    Returns:
        String explanation of the match
    """
    explanations = []

    # Color match explanation
    prefs = user_profile.get("preferences", {})
    favorite_colors = prefs.get("favorite_colors", [])
    product_color = product.get("baseColour", "")

    if product_color.lower() in [c.lower() for c in favorite_colors]:
        explanations.append(f"Matches your favorite color ({product_color})")

    # Event fit explanation
    if event and scores.get("event_fit", 0) >= 0.5:
        event_type = event.get("event_type", "event")
        explanations.append(f"Perfect for {event_type}")

    # Style match explanation
    if scores.get("style_match", 0) >= 0.5:
        style_prefs = get_style_preferences(prefs)
        if style_prefs:
            explanations.append(f"Matches your {style_prefs[0]} style")

    # Trending explanation
    if scores.get("trending", 0) >= 0.7:
        explanations.append("Currently trending")

    # New arrival explanation
    if product.get("is_new_arrival", False):
        explanations.append("New arrival")

    # Stock explanation
    stock = product.get("stock_quantity", 0)
    if stock <= 5 and stock > 0:
        explanations.append(f"Only {stock} left in stock!")

    if not explanations:
        explanations.append("Great option within your budget")

    return " • ".join(explanations)


# =============================================================================
# OPENAI AGENTS SDK - FUNCTION TOOLS
# =============================================================================

@function_tool
def search_products(query: str, category: str = None) -> str:
    """
    Search for products matching the user's query.

    Args:
        query: Natural language search query describing what the user wants
        category: Optional category filter (e.g., 'Dress', 'Shirt', 'Heels')

    Returns:
        JSON string with list of matching products
    """
    # Get user profile from session state
    user_profile = st.session_state.get("user_profile", {})
    if not user_profile:
        return "Error: No user profile selected. Please select a user first."

    # Search products using context-aware pipeline
    results = search_products_with_context(query, user_profile, limit=5)

    # Apply category filter if provided
    if category and results:
        results = [p for p in results if p.get("articleType", "").lower() == category.lower()]

    if not results:
        st.session_state.last_search_results = []
        st.session_state.search_counter += 1
        return "No products found matching your criteria."

    formatted = format_products_for_ui(results)
    st.session_state.last_search_results = formatted
    st.session_state.search_counter += 1

    import json
    return json.dumps(formatted, indent=2)


@function_tool
def add_to_cart(product_id: str, quantity: int = 1) -> str:
    """
    Add a product to the user's shopping cart.

    Args:
        product_id: The unique identifier of the product to add
        quantity: Number of items to add (default: 1)

    Returns:
        Confirmation message with updated cart info
    """
    # Fetch the product
    product = get_product_by_id(product_id)
    if not product:
        return f"Error: Product with ID {product_id} not found."

    # Check stock
    stock = _coerce_int(product.get("stock_quantity", 0))
    if stock < quantity:
        return f"Error: Only {stock} items available in stock."

    # Add to cart in session state
    cart = st.session_state.get("cart", [])

    # Check if product already in cart
    for item in cart:
        if item["product_id"] == product_id:
            item["quantity"] += quantity
            st.session_state.cart = cart
            return f"Updated quantity for {product.get('productDisplayName', 'item')}. Now have {item['quantity']} in cart."

    # Add new item to cart
    cart_item = {
        "product_id": product_id,
        "name": product.get("productDisplayName", "Unknown"),
        "price": _coerce_float(product.get("price", 0)),
        "quantity": quantity,
        "color": product.get("baseColour", ""),
        "type": product.get("articleType", "")
    }
    cart.append(cart_item)
    st.session_state.cart = cart

    # Calculate cart total
    total = sum(item["price"] * item["quantity"] for item in cart)

    return f"Added {product.get('productDisplayName', 'item')} to cart! Cart total: ${total:.2f} ({len(cart)} items)"


@function_tool
def get_accessories(product_id: str) -> str:
    """
    Get matching accessories/complementary items for a product.

    Args:
        product_id: The product ID to find accessories for

    Returns:
        JSON string with list of complementary products
    """
    # Fetch the main product
    product = get_product_by_id(product_id)
    if not product:
        return f"Error: Product with ID {product_id} not found."

    # Get complementary products
    accessories = get_complementary_products(product, limit=3)

    if not accessories:
        return "No matching accessories found."

    # Format results
    formatted = []
    for item in accessories:
        formatted.append({
            "id": str(item.get("_id", "")),
            "name": item.get("productDisplayName", "Unknown"),
            "price": item.get("price", 0),
            "color": item.get("baseColour", ""),
            "type": item.get("articleType", "")
        })

    import json
    return json.dumps(formatted, indent=2)


@function_tool
def view_cart() -> str:
    """
    View the current shopping cart contents.

    Returns:
        JSON string with cart items and total
    """
    cart = st.session_state.get("cart", [])

    if not cart:
        return "Your cart is empty."

    total = sum(item["price"] * item["quantity"] for item in cart)

    import json
    return json.dumps({
        "items": cart,
        "total": total,
        "item_count": len(cart)
    }, indent=2)


# =============================================================================
# OPENAI AGENTS SDK - AGENT DEFINITION
# =============================================================================

def build_system_prompt(user_profile: dict) -> str:
    """Build personalized system prompt with user context."""
    name = user_profile.get("name", "Customer")
    email = user_profile.get("email", "")

    # Get preferences
    prefs = user_profile.get("preferences", {})
    style_prefs = get_style_preferences(prefs)
    colors = prefs.get("favorite_colors", [])
    price_range = prefs.get("price_range", {})
    min_price = price_range.get("min", 0)
    max_price = price_range.get("max", 500)

    # Get sizes
    sizes = user_profile.get("sizes", {})
    size_top = get_size_value(sizes, ["top", "tops"])
    size_bottom = get_size_value(sizes, ["bottom", "bottoms"])
    size_dress = get_size_value(sizes, ["dress", "dresses"])
    size_shoes = get_size_value(sizes, ["shoes", "shoe"])

    # Get upcoming event
    events = user_profile.get("upcoming_events", [])
    event_context = ""
    if events:
        event = events[0]
        event_name = event.get("event_type", "event")
        event_date = event.get("date", "")
        dress_code = event.get("dress_code", "")
        venue = event.get("venue", "")
        notes = event.get("notes", "")

        # Calculate days until event
        if event_date:
            try:
                event_dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                days_until = (event_dt - datetime.now(event_dt.tzinfo)).days
                days_str = f"{days_until} days away" if days_until > 0 else "today"
            except (ValueError, TypeError):
                days_str = ""

            event_context = f"""
Upcoming Event:
- Type: {event_name}
- Date: {event_date} ({days_str})
- Dress Code: {dress_code}
- Venue: {venue}
- Notes: {notes}
"""

    return f"""You are a personal shopping assistant for RetailNext, an AI-powered retail platform.

User Profile:
- Name: {name}
- Email: {email}
- Style Preferences: {', '.join(style_prefs) if style_prefs else 'Not specified'}
- Favorite Colors: {', '.join(colors) if colors else 'Not specified'}
- Budget Range: ${min_price} - ${max_price}
- Sizes: Top: {size_top}, Bottom: {size_bottom}, Dress: {size_dress}, Shoes: {size_shoes}
{event_context}

Instructions:
- Be helpful, friendly, and conversational
- Reference the user's preferences naturally in your responses
- When showing products, explain WHY each one matches their needs
- Suggest accessories when appropriate (after they choose a main item)
- Keep responses concise but personalized
- Always use the search_products tool when the user asks for product recommendations
- Use add_to_cart when the user wants to add items
- Use get_accessories to suggest complementary items
- Use view_cart to show current cart contents

Remember: You're helping {name} find the perfect items for their needs!"""


def create_shopping_agent(user_profile: dict) -> Agent:
    """Create a shopping assistant agent with user context."""
    system_prompt = build_system_prompt(user_profile)

    agent = Agent(
        name="RetailNext Shopping Assistant",
        instructions=system_prompt,
        tools=[search_products, add_to_cart, get_accessories, view_cart],
        model="gpt-4o"
    )

    return agent


def run_agent_sync(agent: Agent, user_message: str, conversation_history: list = None) -> tuple[str, list]:
    """
    Run the agent synchronously and return the response.

    Args:
        agent: The shopping agent instance
        user_message: The user's message
        conversation_history: Previous messages for context

    Returns:
        Tuple of (response text, tool-derived products list)
    """
    # Build input messages with history
    messages = []

    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Run the agent synchronously
    result = Runner.run_sync(agent, messages)

    # Extract the final response
    if result and result.final_output:
        response_text = result.final_output
    else:
        response_text = "I apologize, but I couldn't process your request. Please try again."

    tool_products = extract_products_from_run(result)
    return response_text, tool_products


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "cart" not in st.session_state:
        st.session_state.cart = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = []
    if "search_counter" not in st.session_state:
        st.session_state.search_counter = 0

# Initialize session state
init_session_state()

# =============================================================================
# PLACEHOLDER FOR MAIN APPLICATION
# =============================================================================

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Profile card styling */
    .profile-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Event card styling */
    .event-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }

    /* Product card styling */
    .product-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }

    /* Chat message styling */
    .user-message {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .assistant-message {
        background-color: #f5f5f5;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    /* Badge styling */
    .badge-new {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
    }

    .badge-low-stock {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
    }

    /* Score bar styling */
    .score-bar {
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        background-color: #007bff;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with user profile, event, and cart."""
    with st.sidebar:
        st.markdown("## 👤 Select User")

        # Fetch all users for dropdown
        users = get_all_users()
        if not users:
            st.error("No users found in database")
            return

        # Create user options for selectbox
        user_options = {f"{u['name']} ({u['email']})": u["_id"] for u in users}
        user_names = list(user_options.keys())

        # Find current selection index
        current_index = 0
        if st.session_state.current_user_id:
            for i, u in enumerate(users):
                if u["_id"] == st.session_state.current_user_id:
                    current_index = i
                    break

        # User selector
        selected_name = st.selectbox(
            "Choose a demo user:",
            user_names,
            index=current_index,
            key="user_selector"
        )

        selected_user_id = user_options[selected_name]

        # Handle user change
        if selected_user_id != st.session_state.current_user_id:
            # Save current conversation before switching (if any)
            if st.session_state.current_user_id and st.session_state.conversation_history:
                save_conversation(
                    st.session_state.current_user_id,
                    st.session_state.session_id,
                    st.session_state.conversation_history
                )

            # Update state
            st.session_state.current_user_id = selected_user_id
            st.session_state.user_profile = get_user_profile(selected_user_id)
            st.session_state.cart = []
            st.session_state.display_messages = []
            st.session_state.conversation_history = []
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.agent = create_shopping_agent(st.session_state.user_profile)
            st.rerun()

        st.divider()

        # Profile Card
        if st.session_state.user_profile:
            profile = st.session_state.user_profile
            prefs = profile.get("preferences", {})

            st.markdown("### Profile")

            # Name and email
            st.markdown(f"**{profile.get('name', 'Unknown')}**")
            st.caption(profile.get("email", ""))

            # Style preferences
            style_prefs = get_style_preferences(prefs)
            if style_prefs:
                st.markdown("**Style:**")
                cols = st.columns(len(style_prefs[:3]))
                for i, style in enumerate(style_prefs[:3]):
                    cols[i].markdown(f"`{style}`")

            # Favorite colors
            colors = prefs.get("favorite_colors", [])
            if colors:
                st.markdown("**Colors:**")
                color_text = " • ".join(colors)
                st.markdown(f"🎨 {color_text}")

            # Budget range
            price_range = prefs.get("price_range", {})
            if price_range:
                min_p = price_range.get("min", 0)
                max_p = price_range.get("max", 500)
                st.markdown(f"**Budget:** ${min_p} - ${max_p}")

            # Sizes
            sizes = profile.get("sizes", {})
            if sizes:
                st.markdown("**Sizes:**")
                size_text = (
                    f"Top: {get_size_value(sizes, ['top', 'tops'])} | "
                    f"Bottom: {get_size_value(sizes, ['bottom', 'bottoms'])} | "
                    f"Shoes: {get_size_value(sizes, ['shoes', 'shoe'])}"
                )
                st.caption(size_text)

            st.divider()

            # Upcoming Event Card
            events = profile.get("upcoming_events", [])
            if events:
                event = events[0]
                st.markdown("### 📅 Upcoming Event")

                # Event card with styling
                event_type = event.get("event_type", "Event").title()
                event_date = event.get("date", "")
                dress_code = event.get("dress_code", "Not specified")
                venue = event.get("venue", "Not specified")
                notes = event.get("notes", "")

                # Calculate days until event
                days_str = ""
                if event_date:
                    try:
                        event_dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                        days_until = (event_dt.replace(tzinfo=None) - datetime.now()).days
                        if days_until > 0:
                            days_str = f"⏰ {days_until} days away"
                        elif days_until == 0:
                            days_str = "⏰ Today!"
                        else:
                            days_str = "📌 Past event"
                    except (ValueError, TypeError):
                        pass

                st.markdown(f"**{event_type}**")
                if days_str:
                    st.markdown(days_str)
                st.markdown(f"📍 {venue}")
                st.markdown(f"👔 {dress_code}")
                if notes:
                    st.caption(f"📝 {notes}")

                st.divider()

        # Shopping Cart
        render_cart_sidebar()


def render_cart_sidebar():
    """Render the shopping cart in sidebar."""
    st.markdown("### 🛒 Shopping Cart")

    cart = st.session_state.get("cart", [])

    if not cart:
        st.info("Your cart is empty")
        return

    # Display cart items
    for i, item in enumerate(cart):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{item['name'][:30]}...**" if len(item['name']) > 30 else f"**{item['name']}**")
            st.caption(f"${_coerce_float(item['price']):.2f} x {item['quantity']}")
        with col2:
            if st.button("🗑️", key=f"remove_{i}"):
                st.session_state.cart.pop(i)
                st.rerun()

    # Cart total
    total = sum(_coerce_float(item["price"]) * item["quantity"] for item in cart)
    st.divider()
    st.markdown(f"**Total: ${total:.2f}**")
    st.caption(f"{len(cart)} item(s)")

    # Checkout button
    if st.button("🛍️ Checkout", type="primary", use_container_width=True):
        st.success("Order placed successfully! (Demo)")
        st.session_state.cart = []
        st.rerun()


# =============================================================================
# MAIN CHAT AREA UI
# =============================================================================

def render_chat_area():
    """Render the main chat interface."""
    st.markdown("## 💬 Chat with your Shopping Assistant")

    # Check if user is selected
    if not st.session_state.user_profile:
        st.info("👈 Please select a user from the sidebar to start shopping!")
        return

    # Display user greeting
    user_name = st.session_state.user_profile.get("name", "").split()[0]
    events = st.session_state.user_profile.get("upcoming_events", [])

    if not st.session_state.display_messages:
        # Show welcome message
        welcome_msg = f"Hi {user_name}! 👋 I'm your personal shopping assistant. "
        if events:
            event = events[0]
            event_type = event.get("event_type", "event")
            welcome_msg += f"I see you have a {event_type} coming up! How can I help you find the perfect outfit?"
        else:
            welcome_msg += "How can I help you find something special today?"

        st.session_state.display_messages.append({
            "role": "assistant",
            "content": welcome_msg
        })

    # Display chat messages
    for message in st.session_state.display_messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

            # Display products if included in message
            products = message.get("products", [])
            if products:
                render_product_cards(products)

    # Chat input
    if user_input := st.chat_input("Ask me about dresses, accessories, or anything else..."):
        # Add user message to display
        st.session_state.display_messages.append({
            "role": "user",
            "content": user_input
        })

        # Add to conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        def looks_like_product_query(text: str) -> bool:
            keywords = [
                "dress", "dresses", "gown", "shirt", "top", "blouse", "shoes", "heels",
                "pants", "trousers", "jeans", "skirt", "kurta", "saree", "sari",
                "outfit", "recommend", "suggest", "find", "looking for"
            ]
            text_lower = text.lower()
            return any(k in text_lower for k in keywords)

        def response_denies_results(text: str) -> bool:
            deny_phrases = [
                "couldn't find", "could not find", "no products", "no dresses",
                "no items", "not available", "not in stock"
            ]
            text_lower = text.lower()
            return any(p in text_lower for p in deny_phrases)

        def build_product_summary(products: list) -> str:
            lines = ["Here are a few options I found:"]
            for p in products[:3]:
                name = p.get("name", "Unknown")
                price = _coerce_float(p.get("price", 0))
                color = p.get("color", "")
                extra = f" ({color})" if color else ""
                lines.append(f"- {name}{extra} — ${price:.2f}")
            return "\n".join(lines)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                search_counter_before = st.session_state.search_counter
                st.session_state.last_search_results = []

                # Create agent if not exists
                if st.session_state.agent is None:
                    st.session_state.agent = create_shopping_agent(st.session_state.user_profile)

                # Run agent
                response, tool_products = run_agent_sync(
                    st.session_state.agent,
                    user_input,
                    st.session_state.conversation_history
                )

            products = tool_products or []
            if not products and st.session_state.search_counter > search_counter_before:
                products = st.session_state.get("last_search_results", [])
            if not products and looks_like_product_query(user_input):
                # Fallback: run search directly if the agent didn't call the tool
                fallback_results = search_products_with_context(
                    user_input,
                    st.session_state.user_profile,
                    limit=5
                )
                products = format_products_for_ui(fallback_results)
                st.session_state.last_search_results = products

            if products and not st.session_state.last_search_results:
                st.session_state.last_search_results = products

            # If we actually found products but the agent says none, override response
            if products and response_denies_results(response):
                response = build_product_summary(products)

            st.markdown(response)
            if products:
                render_product_cards(products)

        st.session_state.display_messages.append({
            "role": "assistant",
            "content": response,
            "products": products
        })

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        st.rerun()


def render_product_cards(products: list):
    """Render product recommendation cards."""
    if not products:
        return

    cols = st.columns(min(len(products), 3))

    for i, product in enumerate(products[:3]):
        with cols[i]:
            # Product card container
            with st.container():
                # Product image placeholder
                article_type = product.get("type", product.get("articleType", "Item"))
                color = product.get("color", product.get("baseColour", ""))
                image_url = product.get("image_url")

                # Icon based on article type
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
                icon = icons.get(article_type, "👕")

                if image_url:
                    st.image(image_url, use_column_width=True)
                else:
                    st.markdown(f"### {icon}")

                # Product name
                name = product.get("name", product.get("productDisplayName", "Unknown"))
                st.markdown(f"**{name[:40]}**" if len(name) > 40 else f"**{name}**")

                # Price
                price = _coerce_float(product.get("price", 0))
                st.markdown(f"💰 **${price:.2f}**")

                # Color
                if color:
                    st.caption(f"Color: {color}")

                # Stock badge
                stock = _coerce_int(product.get("stock", product.get("stock_quantity", 0)))
                if stock <= 5 and stock > 0:
                    st.markdown("🔴 **Low Stock!**")
                elif product.get("is_new", product.get("is_new_arrival", False)):
                    st.markdown("🟢 **New Arrival**")

                # Match score if available
                score = product.get("score", 0)
                if score > 0:
                    st.progress(score, text=f"Match: {score*100:.0f}%")

                # Add to cart button
                product_id = str(product.get("id", product.get("_id", "")))
                if st.button(f"Add to Cart", key=f"add_{product_id}_{i}"):
                    add_to_cart(product_id, 1)
                    st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def render_debug_panel():
    """Render debug information panel."""
    with st.expander("🔧 Debug Panel", expanded=False):
        st.markdown("### Search Debug Log")

        debug_log = st.session_state.get("debug_log", [])

        if not debug_log:
            st.info("No searches performed yet.")
            return

        # Show last 5 searches
        for i, log in enumerate(reversed(debug_log[-5:])):
            st.markdown(f"**Search #{len(debug_log) - i}**")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Query:**")
                st.code(log.get("original_query", "N/A"))
                st.markdown("**Transformed Query:**")
                st.code(log.get("transformed_query", "N/A"))

            with col2:
                st.markdown("**Filters:**")
                st.code(log.get("filters", "N/A"))
                st.markdown(f"**Embedding:** {log.get('embedding_dimensions', 0)} dims")
                st.markdown(f"**Results Count:** {log.get('results_count', 0)}")
                if log.get("fallback_attempts"):
                    st.markdown("**Fallbacks:**")
                    st.code(log.get("fallback_attempts"))

            # Show results preview
            results_preview = log.get("results_preview", [])
            if results_preview:
                st.markdown("**Top Results:**")
                for r in results_preview:
                    st.caption(f"• {r['name']} (score: {r['score']:.4f})")
            else:
                st.warning("No results returned")

            st.divider()

        # Clear debug log button
        if st.button("Clear Debug Log"):
            st.session_state.debug_log = []
            st.rerun()


def main():
    """Main application entry point."""
    # Header
    st.title(f"{APP_ICON} {APP_TITLE}")

    # Check database connection
    if db is None:
        st.error("Database connection failed. Please check your MongoDB URI.")
        return

    if openai_client is None:
        st.error("OpenAI client initialization failed. Please check your API key.")
        return

    # Render sidebar
    render_sidebar()

    # Render main chat area
    render_chat_area()

    # Render debug panel at the bottom
    render_debug_panel()


if __name__ == "__main__":
    main()
