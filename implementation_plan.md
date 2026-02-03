# RetailNext Intelligent Search - Simple Implementation Plan

## Goal
Implement a minimal demo with only two behaviors:
1. Use raw vector search to find products for user keywords.
2. Show the results in the chat UI.

## Scope (Only These Features)
- Query → embedding → vector search
- Render top results in chat (name, price, image)
- No agent, no personalization, no cart, no events

## Data Dependencies
- Collection: `productAndEmbeddings`
- Vector index: `vector_index`
- Embedding field: `embeddings`
- Required fields: `productDisplayName`, `price`, `image_url` (or image path)

## Minimal Flow
1. User enters a query in chat.
2. Generate embedding for the raw user query (no transformation).
3. Execute MongoDB Atlas vector search with the embedding.
4. Render top 3 products in the chat UI.

## Pseudocode
```python
def search_products_raw(query: str) -> list:
    embedding = generate_embedding(query)
    return vector_search_products(embedding, limit=3)

def render_chat():
    if user_input := st.chat_input(...):
        results = search_products_raw(user_input)
        st.chat_message("assistant").markdown("Here are the top matches:")
        render_product_cards(results)
```

## Success Criteria
- Query “red dress” returns products.
- Chat UI shows product cards with name + price (+ image if available).
