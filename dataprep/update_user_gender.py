"""Update user_profiles with gender based on first name rules."""

import os
import argparse
import certifi
from pymongo import MongoClient

DEFAULT_DB = "retail_demo"
COLLECTION = "user_profiles"

FEMALE_NAMES = {"Sarah", "Emma", "Priya", "Jordan", "Alex", "Jennifer"}
MALE_NAMES = {"Michael", "David", "Tyler", "Robert"}


def parse_args():
    parser = argparse.ArgumentParser(description="Set gender fields in user profiles.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--default", default="Unisex", help="Fallback gender if name not matched")
    parser.add_argument("--field", default="gender", help="Field to write (e.g., gender or gender_preference)")
    parser.add_argument("--also-set", default="gender_preference", help="Secondary field to also write")
    return parser.parse_args()


def infer_gender(name: str, default: str) -> str:
    if not name:
        return default
    first = name.split()[0]
    if first in FEMALE_NAMES:
        return "Women"
    if first in MALE_NAMES:
        return "Men"
    return default


def main():
    args = parse_args()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI not set")

    client = MongoClient(uri, tlsCAFile=certifi.where())
    db = client[args.db]
    col = db[args.collection]

    cursor = col.find({}, {"_id": 1, "name": 1})

    updated = 0
    total = 0
    for doc in cursor:
        total += 1
        name = doc.get("name", "")
        gender = infer_gender(name, args.default)

        if args.dry_run:
            if total <= 10:
                print(f"Would set {args.field}={gender}, {args.also_set}={gender} for {doc.get('_id')} ({name})")
            continue

        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {args.field: gender, args.also_set: gender}}
        )
        updated += 1

    print(f"Processed: {total}, Updated: {updated}")


if __name__ == "__main__":
    main()
