# Repository Guidelines

## Purpose & Demo Story
RetailNext is a **conversational retail demo** that highlights a business UX plus a parallel technical view. The primary story is a **job interview outfit search**, but the flow supports other events (wedding, business trip, etc.). The Chat tab shows the customer experience; the **Tech Demo** tab shows JSON for each system step.

## Core Workflow (Deterministic)
1. Extract intent (category + color + event) with `gpt-4o-mini`.
2. Vector search (top 5) via MongoDB Atlas using embeddings.
3. Rerank (top 3) with `gpt-4o` using event context + product images.
4. Generate a friendly summary and recommendations.
5. Actions: stock check → add to cart → payment → delivery message.
6. Optional virtual try‑on via `gpt-image-1.5`.

## Models & Services
- Intent: `gpt-4o-mini`
- Embeddings: `text-embedding-3-large`
- Rerank + Summary: `gpt-4o`
- Try‑On: `gpt-image-1.5`
- Vector search: MongoDB Atlas (`vector_index` on `embeddings`)

## Data Dependencies
Database: `retail_demo`
Collections: `productAndEmbeddings`, `user_profiles`, `user_conversation_memory`

## UX Rules
- Ask for missing **color** if not provided.
- Event context affects **rerank only**, not vector search.
- Chat must never render raw JSON.
- Tech Demo tab shows JSON blocks per step with short explanations.

## Development Commands
- `pip install -r requirements.txt`
- `streamlit run app.py`
- `python -m py_compile app.py`

## Data Prep
- `python dataprep/update_image_urls.py`
- `python dataprep/update_user_gender.py`

## Security
- Never commit `.env` or API keys.
- Required env vars: `OPENAI_API_KEY`, `MONGODB_URI`.
