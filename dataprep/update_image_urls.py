"""Update product docs with GitHub raw image URLs based on product id."""

import os
import argparse
from pymongo import MongoClient
import certifi

DEFAULT_DB = "retail_demo"
DEFAULT_COLLECTION = "productAndEmbeddings"
RAW_BASE = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images"


def build_url(image_id: str) -> str:
    return f"{RAW_BASE}/{image_id}.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Attach image_url to products by id.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--id-field", default="id", help="Field name that matches image filename (e.g., id)")
    parser.add_argument("--image-field", default="image_url", help="Field to write URL into")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents processed (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Print sample updates without writing")
    return parser.parse_args()


def main():
    args = parse_args()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI not set")

    client = MongoClient(uri, tlsCAFile=certifi.where())
    db = client[args.db]
    col = db[args.collection]

    query = {args.id_field: {"$exists": True}}
    cursor = col.find(query, {args.id_field: 1, args.image_field: 1})

    if args.limit and args.limit > 0:
        cursor = cursor.limit(args.limit)

    updates = 0
    skipped = 0

    for doc in cursor:
        image_id = doc.get(args.id_field)
        if image_id is None:
            skipped += 1
            continue

        url = build_url(str(image_id))

        if args.dry_run:
            if updates < 5:
                print(f"Would set {args.image_field} for {doc.get('_id')} -> {url}")
            updates += 1
            continue

        result = col.update_one(
            {"_id": doc["_id"]},
            {"$set": {args.image_field: url}}
        )
        updates += result.modified_count

    print(f"Processed: {updates + skipped}, Updated: {updates}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
