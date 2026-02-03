import os
import random
import certifi
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Database name to target
TARGET_DATABASE = "retail_demo"

# Price ranges by category (min, max)
PRICE_RANGES = {
    "accessories": (19.99, 79.99),
    "tops": (29.99, 149.99),
    "bottoms": (29.99, 149.99),
    "dresses": (49.99, 299.99),
    "shoes": (39.99, 199.99),
    "outerwear": (79.99, 399.99),
}

# Stock quantity distribution
STOCK_CHOICES = [0, 1, 2, 5, 10, 15, 20, 30]
STOCK_WEIGHTS = [5, 5, 10, 15, 20, 20, 15, 10]


def connect_to_mongodb():
    """Connect to MongoDB Atlas and return the client."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    # Create a MongoDB client with SSL certificate verification
    client = MongoClient(uri, tlsCAFile=certifi.where())
    return client


def generate_sku(gender, category, product_id):
    """Generate SKU in format: RN-{GENDER}-{CATEGORY}-{ID:06d}"""
    # Normalize gender and category to uppercase
    gender_code = gender.upper() if gender else "UNISEX"
    category_code = category.upper().replace(" ", "-") if category else "GENERAL"
    return f"RN-{gender_code}-{category_code}-{product_id:06d}"


def generate_price(category):
    """Generate a realistic price based on category."""
    # Normalize category to lowercase for matching
    category_lower = category.lower() if category else ""

    # Find matching price range
    for key, (min_price, max_price) in PRICE_RANGES.items():
        if key in category_lower:
            return round(random.uniform(min_price, max_price), 2)

    # Default price range if category not found
    return round(random.uniform(29.99, 149.99), 2)


def generate_stock_quantity():
    """Generate stock quantity from weighted distribution."""
    return random.choices(STOCK_CHOICES, weights=STOCK_WEIGHTS, k=1)[0]


def generate_date_added():
    """Generate a random date within the last 180 days."""
    days_ago = random.randint(0, 180)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


def is_new_arrival(date_added_str):
    """Check if product is a new arrival (added within last 30 days)."""
    date_added = datetime.strptime(date_added_str, "%Y-%m-%d")
    days_since_added = (datetime.now() - date_added).days
    return days_since_added <= 30


def generate_popularity_score():
    """Generate popularity score using normal distribution centered at 0.7."""
    # Normal distribution with mean=0.7, std=0.15, clamped to [0.3, 1.0]
    score = random.gauss(0.7, 0.15)
    return round(max(0.3, min(1.0, score)), 2)


def add_fields_to_products(client):
    """Add new fields to all products in the Retail Demo database."""
    db = client[TARGET_DATABASE]
    collections = db.list_collection_names()

    if not collections:
        print(f"No collections found in '{TARGET_DATABASE}'")
        return

    print(f"\nProcessing database: {TARGET_DATABASE}")
    print("=" * 50)

    total_updated = 0

    for collection_name in collections:
        print(f"\n  Collection: {collection_name}")
        print(f"  {'-' * 40}")

        collection = db[collection_name]
        documents = list(collection.find())

        if not documents:
            print("    No documents found")
            continue

        updated_count = 0

        for idx, doc in enumerate(documents):
            # Extract gender and category from existing document fields
            gender = doc.get("gender", doc.get("Gender", ""))
            category = doc.get("category", doc.get("Category", ""))

            # Generate unique product ID based on document index
            product_id = idx + 1

            # Generate all new fields
            date_added = generate_date_added()

            new_fields = {
                "sku": generate_sku(gender, category, product_id),
                "price": generate_price(category),
                "stock_quantity": generate_stock_quantity(),
                "date_added": date_added,
                "is_new_arrival": is_new_arrival(date_added),
                "popularity_score": generate_popularity_score(),
            }

            # Update the document in MongoDB
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": new_fields}
            )

            updated_count += 1
            print(f"    Updated: {doc.get('name', doc.get('_id'))} -> SKU: {new_fields['sku']}")

        print(f"    Total updated: {updated_count} document(s)")
        total_updated += updated_count

    print(f"\n{'=' * 50}")
    print(f"Total products updated: {total_updated}")


if __name__ == "__main__":
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB Atlas...")
        client = connect_to_mongodb()

        # Test the connection
        client.admin.command("ping")
        print("Successfully connected to MongoDB Atlas!")

        # Verify target database exists
        db_names = client.list_database_names()
        if TARGET_DATABASE not in db_names:
            print(f"\nError: Database '{TARGET_DATABASE}' not found.")
            print(f"Available databases: {db_names}")
        else:
            # Add fields to products
            add_fields_to_products(client)

        # Close the connection
        client.close()
        print("\nConnection closed.")

    except Exception as e:
        print(f"Error: {e}")
