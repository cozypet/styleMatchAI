import os
import certifi
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Database and collection names
TARGET_DATABASE = "retail_demo"
COLLECTION_NAME = "user_profiles"


def connect_to_mongodb():
    """Connect to MongoDB Atlas and return the client."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    # Create a MongoDB client with SSL certificate verification
    client = MongoClient(uri, tlsCAFile=certifi.where())
    return client


def generate_random_date(days_back_min, days_back_max):
    """Generate a random ISO datetime within a range of days back from now."""
    days_ago = random.randint(days_back_min, days_back_max)
    hours_ago = random.randint(0, 23)
    date = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
    return date.isoformat()


def generate_user_profiles():
    """Generate 10 user profiles with diverse personas."""

    profiles = [
        # Demo Scenario 1: Wedding Guest - Sarah Chen
        {
            "_id": "user_001",
            "name": "Sarah Chen",
            "email": "sarah.chen@email.com",
            "preferences": {
                "style": ["elegant", "feminine", "sophisticated"],
                "favorite_colors": ["blush pink", "sage green", "champagne", "navy"],
                "price_range": {"min": 100, "max": 400},
                "items_to_avoid": ["black dresses", "white dresses", "overly casual", "too revealing"],
                "preferred_brands": ["Anthropologie", "BHLDN", "Reformation"],
                "fit_preference": "fitted but comfortable"
            },
            "sizes": {
                "tops": "S",
                "dresses": "4",
                "bottoms": "26",
                "shoes": "7"
            },
            "upcoming_events": [
                {
                    "event_type": "wedding",
                    "event_name": "Best Friend's Garden Wedding",
                    "date": "2026-03-15",
                    "dress_code": "semi-formal",
                    "venue": "outdoor garden",
                    "notes": "Spring wedding, need something flowy and elegant"
                }
            ],
            "created_at": generate_random_date(180, 365),
            "last_active": generate_random_date(0, 2)
        },

        # Demo Scenario 2: Job Interview - Michael Rodriguez
        {
            "_id": "user_002",
            "name": "Michael Rodriguez",
            "email": "m.rodriguez@email.com",
            "preferences": {
                "style": ["professional", "modern", "polished"],
                "favorite_colors": ["navy", "charcoal", "white", "light blue"],
                "price_range": {"min": 150, "max": 500},
                "items_to_avoid": ["loud patterns", "casual wear", "sneakers"],
                "preferred_brands": ["Hugo Boss", "Brooks Brothers", "Bonobos"],
                "fit_preference": "slim fit"
            },
            "sizes": {
                "tops": "M",
                "suits": "40R",
                "bottoms": "32x32",
                "shoes": "10"
            },
            "upcoming_events": [
                {
                    "event_type": "job_interview",
                    "event_name": "Senior Product Manager Interview at Tech Company",
                    "date": "2026-02-18",
                    "dress_code": "business professional",
                    "venue": "corporate office",
                    "notes": "Tech company, want to look sharp but not overly formal"
                }
            ],
            "created_at": generate_random_date(90, 180),
            "last_active": generate_random_date(0, 1)
        },

        # Demo Scenario 3: Date Night - Emma Thompson
        {
            "_id": "user_003",
            "name": "Emma Thompson",
            "email": "emma.t@email.com",
            "preferences": {
                "style": ["chic", "romantic", "trendy"],
                "favorite_colors": ["red", "black", "burgundy", "emerald"],
                "price_range": {"min": 75, "max": 250},
                "items_to_avoid": ["overly casual", "athletic wear", "baggy clothes"],
                "preferred_brands": ["Zara", "Revolve", "Aritzia"],
                "fit_preference": "form-fitting"
            },
            "sizes": {
                "tops": "XS",
                "dresses": "2",
                "bottoms": "25",
                "shoes": "6.5"
            },
            "upcoming_events": [
                {
                    "event_type": "date_night",
                    "event_name": "Anniversary Dinner",
                    "date": "2026-02-14",
                    "dress_code": "smart casual to dressy",
                    "venue": "upscale restaurant",
                    "notes": "2 year anniversary, want to look stunning but effortless"
                }
            ],
            "created_at": generate_random_date(60, 120),
            "last_active": generate_random_date(0, 1)
        },

        # Additional Persona 4: Business Trip - David Park
        {
            "_id": "user_004",
            "name": "David Park",
            "email": "david.park@corporate.com",
            "preferences": {
                "style": ["classic", "versatile", "minimalist"],
                "favorite_colors": ["black", "gray", "navy", "white"],
                "price_range": {"min": 100, "max": 350},
                "items_to_avoid": ["wrinkle-prone fabrics", "high maintenance items"],
                "preferred_brands": ["Uniqlo", "Theory", "COS"],
                "fit_preference": "regular fit"
            },
            "sizes": {
                "tops": "L",
                "suits": "42R",
                "bottoms": "34x30",
                "shoes": "11"
            },
            "upcoming_events": [
                {
                    "event_type": "business_trip",
                    "event_name": "Client Meetings in New York",
                    "date": "2026-02-25",
                    "dress_code": "business casual",
                    "venue": "multiple venues",
                    "notes": "Need versatile pieces that pack well for 3-day trip"
                },
                {
                    "event_type": "networking",
                    "event_name": "Industry Mixer",
                    "date": "2026-02-26",
                    "dress_code": "smart casual",
                    "venue": "rooftop bar",
                    "notes": "Evening networking event during the trip"
                }
            ],
            "created_at": generate_random_date(200, 400),
            "last_active": generate_random_date(1, 5)
        },

        # Additional Persona 5: Vacation - Priya Sharma
        {
            "_id": "user_005",
            "name": "Priya Sharma",
            "email": "priya.sharma@gmail.com",
            "preferences": {
                "style": ["bohemian", "colorful", "relaxed"],
                "favorite_colors": ["turquoise", "coral", "white", "gold"],
                "price_range": {"min": 50, "max": 200},
                "items_to_avoid": ["dark colors", "heavy fabrics", "formal wear"],
                "preferred_brands": ["Free People", "Anthropologie", "Madewell"],
                "fit_preference": "flowy and comfortable"
            },
            "sizes": {
                "tops": "M",
                "dresses": "8",
                "bottoms": "28",
                "shoes": "8"
            },
            "upcoming_events": [
                {
                    "event_type": "vacation",
                    "event_name": "Beach Resort Trip to Bali",
                    "date": "2026-04-10",
                    "dress_code": "resort casual",
                    "venue": "beach resort",
                    "notes": "10-day vacation, need resort wear, swimsuit coverups, evening dinner outfits"
                }
            ],
            "created_at": generate_random_date(30, 90),
            "last_active": generate_random_date(0, 3)
        },

        # Additional Persona 6: Graduation - Tyler Washington
        {
            "_id": "user_006",
            "name": "Tyler Washington",
            "email": "tyler.w@university.edu",
            "preferences": {
                "style": ["preppy", "youthful", "smart"],
                "favorite_colors": ["blue", "khaki", "white", "burgundy"],
                "price_range": {"min": 40, "max": 150},
                "items_to_avoid": ["overly formal", "outdated styles"],
                "preferred_brands": ["J.Crew", "Banana Republic", "Ralph Lauren"],
                "fit_preference": "modern fit"
            },
            "sizes": {
                "tops": "M",
                "suits": "38R",
                "bottoms": "30x32",
                "shoes": "9.5"
            },
            "upcoming_events": [
                {
                    "event_type": "graduation",
                    "event_name": "MBA Graduation Ceremony",
                    "date": "2026-05-20",
                    "dress_code": "business formal under gown",
                    "venue": "university auditorium",
                    "notes": "Family dinner after ceremony, need outfit that works for both"
                }
            ],
            "created_at": generate_random_date(150, 300),
            "last_active": generate_random_date(2, 7)
        },

        # Additional Persona 7: Fitness Enthusiast - Jordan Rivera
        {
            "_id": "user_007",
            "name": "Jordan Rivera",
            "email": "jordan.fit@email.com",
            "preferences": {
                "style": ["athleisure", "sporty", "functional"],
                "favorite_colors": ["black", "neon green", "gray", "coral"],
                "price_range": {"min": 30, "max": 120},
                "items_to_avoid": ["non-breathable fabrics", "restrictive clothing"],
                "preferred_brands": ["Lululemon", "Nike", "Athleta", "Vuori"],
                "fit_preference": "fitted athletic"
            },
            "sizes": {
                "tops": "S",
                "bottoms": "XS",
                "sports_bra": "32C",
                "shoes": "7.5"
            },
            "upcoming_events": [
                {
                    "event_type": "fitness_event",
                    "event_name": "Half Marathon",
                    "date": "2026-03-08",
                    "dress_code": "athletic",
                    "venue": "outdoor race course",
                    "notes": "Need performance running gear and post-race casual outfit"
                }
            ],
            "created_at": generate_random_date(100, 200),
            "last_active": generate_random_date(0, 1)
        },

        # Additional Persona 8: Creative Professional - Alex Kim
        {
            "_id": "user_008",
            "name": "Alex Kim",
            "email": "alex.creative@design.co",
            "preferences": {
                "style": ["avant-garde", "artistic", "edgy"],
                "favorite_colors": ["black", "white", "mustard", "olive"],
                "price_range": {"min": 80, "max": 300},
                "items_to_avoid": ["corporate looks", "mainstream basics"],
                "preferred_brands": ["COS", "Acne Studios", "& Other Stories"],
                "fit_preference": "oversized and architectural"
            },
            "sizes": {
                "tops": "M",
                "dresses": "6",
                "bottoms": "27",
                "shoes": "8"
            },
            "upcoming_events": [
                {
                    "event_type": "art_exhibition",
                    "event_name": "Gallery Opening Night",
                    "date": "2026-02-28",
                    "dress_code": "creative black tie",
                    "venue": "art gallery",
                    "notes": "My work is being featured, want to make a statement"
                }
            ],
            "created_at": generate_random_date(45, 90),
            "last_active": generate_random_date(0, 2)
        },

        # Additional Persona 9: Parent - Jennifer Martinez
        {
            "_id": "user_009",
            "name": "Jennifer Martinez",
            "email": "jen.martinez@family.net",
            "preferences": {
                "style": ["practical", "polished", "mom-chic"],
                "favorite_colors": ["navy", "blush", "cream", "olive"],
                "price_range": {"min": 40, "max": 150},
                "items_to_avoid": ["dry clean only", "white pants", "high heels over 2 inches"],
                "preferred_brands": ["Madewell", "Old Navy", "Target", "Nordstrom"],
                "fit_preference": "comfortable but put-together"
            },
            "sizes": {
                "tops": "L",
                "dresses": "12",
                "bottoms": "31",
                "shoes": "8.5"
            },
            "upcoming_events": [
                {
                    "event_type": "school_event",
                    "event_name": "Parent-Teacher Conference & School Gala",
                    "date": "2026-03-05",
                    "dress_code": "smart casual to semi-formal",
                    "venue": "school gymnasium",
                    "notes": "Conference at 4pm, gala fundraiser at 7pm - need versatile outfit"
                }
            ],
            "created_at": generate_random_date(250, 500),
            "last_active": generate_random_date(0, 4)
        },

        # Additional Persona 10: Retiree - Robert Anderson
        {
            "_id": "user_010",
            "name": "Robert Anderson",
            "email": "r.anderson@retired.com",
            "preferences": {
                "style": ["classic", "comfortable", "timeless"],
                "favorite_colors": ["navy", "tan", "light blue", "forest green"],
                "price_range": {"min": 60, "max": 200},
                "items_to_avoid": ["skinny fits", "trendy items", "complicated closures"],
                "preferred_brands": ["L.L.Bean", "Lands End", "Orvis"],
                "fit_preference": "relaxed classic fit"
            },
            "sizes": {
                "tops": "XL",
                "bottoms": "36x30",
                "shoes": "10.5"
            },
            "upcoming_events": [
                {
                    "event_type": "cruise",
                    "event_name": "Alaska Cruise",
                    "date": "2026-06-15",
                    "dress_code": "resort casual to formal nights",
                    "venue": "cruise ship",
                    "notes": "2-week cruise, need layering options and formal dinner attire"
                },
                {
                    "event_type": "family_reunion",
                    "event_name": "50th Wedding Anniversary Party",
                    "date": "2026-07-04",
                    "dress_code": "garden party attire",
                    "venue": "country club",
                    "notes": "Our golden anniversary celebration, want to look my best"
                }
            ],
            "created_at": generate_random_date(300, 600),
            "last_active": generate_random_date(1, 7)
        }
    ]

    return profiles


def insert_user_profiles(client, profiles):
    """Insert user profiles into MongoDB."""
    db = client[TARGET_DATABASE]
    collection = db[COLLECTION_NAME]

    # Drop existing collection if it exists (for clean regeneration)
    collection.drop()
    print(f"Cleared existing '{COLLECTION_NAME}' collection")

    # Insert all profiles
    result = collection.insert_many(profiles)
    print(f"\nInserted {len(result.inserted_ids)} user profiles into '{COLLECTION_NAME}'")

    return result


def display_profiles(profiles):
    """Display summary of generated profiles."""
    print("\n" + "=" * 60)
    print("GENERATED USER PROFILES SUMMARY")
    print("=" * 60)

    for profile in profiles:
        events = profile.get("upcoming_events", [])
        event_str = ", ".join([e["event_type"] for e in events]) if events else "None"

        print(f"\n{profile['_id']}: {profile['name']}")
        print(f"  Email: {profile['email']}")
        print(f"  Style: {', '.join(profile['preferences']['style'])}")
        print(f"  Price Range: ${profile['preferences']['price_range']['min']} - ${profile['preferences']['price_range']['max']}")
        print(f"  Upcoming Events: {event_str}")


if __name__ == "__main__":
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB Atlas...")
        client = connect_to_mongodb()

        # Test the connection
        client.admin.command("ping")
        print("Successfully connected to MongoDB Atlas!")

        # Generate user profiles
        print("\nGenerating user profiles...")
        profiles = generate_user_profiles()

        # Display profile summary
        display_profiles(profiles)

        # Insert profiles into MongoDB
        print("\n" + "=" * 60)
        print("INSERTING INTO MONGODB")
        print("=" * 60)
        insert_user_profiles(client, profiles)

        # Close the connection
        client.close()
        print("\nConnection closed.")

    except Exception as e:
        print(f"Error: {e}")
