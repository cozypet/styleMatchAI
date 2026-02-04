# RetailNext Intelligent Search - Current Implementation Plan (Aligned)

This plan reflects what is **already implemented** in `app.py` and how the demo behaves today.

## Goal
Deliver a conversational retail demo that:
- Captures user intent (category + color)
- Performs vector search (top 5)
- Reranks with multimodal reasoning (event‑aware)
- Presents top 3 recommendations with clear, human‑readable explanations
- Supports add‑to‑cart, stock check, payment, delivery notice, and virtual try‑on

## Scope (As Implemented)
- ✅ Intent extraction via small LLM (`gpt-4o-mini`), no regex-first expansion
- ✅ Optional event context from user input or upcoming events in profile
- ✅ Vector search (MongoDB Atlas) with gender filter + unisex
- ✅ Rerank step with LLM and image reasoning (event‑first)
- ✅ Step-by-step UI (Step 1/3, Step 2/3, Step 3/3)
- ✅ Clean, conversational summaries + bullet reasons
- ✅ Try‑on image generation via OpenAI Images API
- ✅ Cart + stock check + payment + delivery planned message

## Key Behavior Rules
- **Search query formation**: Only extract **color** + **category** for the vector query.
- **Event handling**:
  - If the user mentions an event (wedding, ceremony, reception, bridal/bridesmaid, groom, party, business trip, etc.), store it and use it for rerank.
  - If no event is mentioned and there is an upcoming event in the profile, ask once if the user wants to use it.
  - Do not nag with repeated yes/no prompts.
- **Gender filtering**:
  - Use explicit user request to override (e.g., “for my husband”).
  - Otherwise, use profile gender or preference.
  - Always include `Unisex` in filter.
- **Results count**:
  - Vector search returns **top 5**.
  - Rerank returns **top 3**.

## Data Dependencies
- MongoDB database: `retail_demo`
- Collections:
  - `productAndEmbeddings`
  - `user_profiles`
  - `user_conversation_memory`
- Vector index: `vector_index` on field `embeddings`
- Product fields used:
  - `productDisplayName`, `price`, `stock_quantity`, `baseColour`, `articleType`, `gender`, `image_url`

## High‑Level Flow
1. **Initial Greeting**
   - Display welcome message: “Hello, what are you looking for today?”

2. **Intent Capture**
   - Extract category + color with `gpt-4o-mini`.
   - If missing color (non‑wedding), ask for color.
   - For wedding/ceremony triggers, ask for **category + color** if missing.

3. **Event Context**
   - If user mentions event, store in session and use for rerank.
   - If no event mentioned but profile has upcoming event, ask once to use it.

4. **Step 1/3: Vector Search**
   - Build query from color + category only.
   - Generate embedding and run Atlas vector search.
   - Show top **5** results in UI.

5. **Step 2/3: Rerank (LLM + Images)**
   - Use LLM to rerank with event‑first reasoning and product images.
   - Keep reasoning concise and human‑readable.

6. **Step 3/3: Recommendations**
   - Show top **3** results with bullet reasons.
   - Use conversational tone, offer to add to cart or check stock.

7. **Post‑Recommendation Actions**
   - Stock check
   - Add to cart
   - Payment success path + delivery planned message

8. **Virtual Try‑On**
   - Trigger from product cards (“Try on”).
   - User can upload a photo, use a URL, or profile photo.
   - Generate try‑on image via OpenAI Images API (`gpt-image-1.5`).

## UX / UI Requirements
- Steps appear sequentially and **remain visible** (not overwritten).
- No raw JSON should be rendered in chat.
- Recommendations are formatted as short paragraphs + bullets.
- Images are shown at reduced size due to low resolution.

## Models (As Implemented)
- Embedding: `text-embedding-3-large`
- Intent: `gpt-4o-mini`
- Rerank/summary: `gpt-4o`
- Try‑on image: `gpt-image-1.5`

## Success Criteria
- Vector search shows 5 items every time results exist.
- Rerank shows 3 items with clear reasons.
- Event context affects rerank, not vector search.
- Conversational output is readable and not JSON.
- Try‑on generates an image with clear error handling.

