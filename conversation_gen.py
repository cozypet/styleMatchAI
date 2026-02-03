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
COLLECTION_NAME = "user_conversation_memory"


def connect_to_mongodb():
    """Connect to MongoDB Atlas and return the client."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    # Create a MongoDB client with SSL certificate verification
    client = MongoClient(uri, tlsCAFile=certifi.where())
    return client


def generate_session_id(base_date):
    """Generate a session ID in format: session_YYYYMMDD_HHMMSS"""
    return f"session_{base_date.strftime('%Y%m%d_%H%M%S')}"


def generate_timestamp(base_date, minutes_offset=0):
    """Generate ISO timestamp with optional minute offset."""
    return (base_date + timedelta(minutes=minutes_offset)).isoformat()


def generate_conversations():
    """Generate conversation histories for all users."""

    conversations = []

    # ===== USER 001: Sarah Chen - Wedding Guest =====
    # Conversation 1: Initial outfit search
    base_date = datetime.now() - timedelta(days=14, hours=10)
    conversations.append({
        "user_id": "user_001",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Hi! I need help finding a dress for my best friend's wedding in March.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "I'd love to help you find the perfect wedding guest dress! I see you have a garden wedding coming up on March 15th. Can you tell me more about the venue and dress code? Is it during the day or evening?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "It's an outdoor garden wedding, semi-formal. The ceremony is at 4pm. I want something elegant but not too stuffy.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Perfect! For a spring garden wedding at 4pm, I'd recommend something flowy and romantic. Based on your preferences for blush pink and sage green, I have some beautiful options. Would you prefer a midi length or floor-length dress?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "I think midi would be better for an outdoor setting. And definitely no white or black!", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Absolutely! I found 3 stunning midi dresses that would be perfect: a sage green wrap dress with flutter sleeves ($189), a blush floral print dress with a romantic silhouette ($225), and a champagne satin slip dress with delicate details ($275). All are in your size 4. Would you like to see them?", "timestamp": generate_timestamp(base_date, 5)},
            {"role": "user", "content": "Yes! The sage green one sounds lovely. Can you show me that one first?", "timestamp": generate_timestamp(base_date, 6)},
            {"role": "assistant", "content": "Here's the Sage Garden Wrap Dress! It features elegant flutter sleeves, a flattering wrap silhouette, and flows beautifully. The fabric is lightweight and perfect for spring. It pairs wonderfully with nude heels or strappy sandals. Would you like me to suggest some complementary accessories?", "timestamp": generate_timestamp(base_date, 7)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 7)
    })

    # Conversation 2: Follow-up on accessories
    base_date = datetime.now() - timedelta(days=10, hours=15)
    conversations.append({
        "user_id": "user_001",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Hey! I decided to go with the sage wrap dress we talked about. Now I need shoes and a bag.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Great choice! The sage dress is beautiful. For a garden wedding, I'd suggest block heels or wedges since you'll be on grass. What's your budget for shoes and a bag?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Maybe $150 total for both? I want something I can wear again.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Smart thinking! I found a nude block heel sandal ($68) that's elegant yet practical for grass, and a gold clutch ($72) that adds a touch of sparkle. Both are versatile pieces you'll reach for again. The total is $140, right in your budget!", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Perfect! What about jewelry? I don't want to overdo it.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "For a sophisticated look, I'd keep jewelry minimal. Simple gold drop earrings would complement without competing with your dress. Maybe a delicate bracelet too. Would you like me to show you some options under $50?", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 3: Day-of questions
    base_date = datetime.now() - timedelta(days=3, hours=9)
    conversations.append({
        "user_id": "user_001",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Quick question - should I bring a wrap or cardigan in case it gets chilly in the evening?", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Great thinking! March evenings can get cool. A lightweight cashmere wrap in a neutral tone would be perfect - elegant enough for the event and practical. I'd recommend ivory or soft gold to complement your sage dress.", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Do you have any in that price range?", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "I found a beautiful ivory pashmina wrap for $45 that would be ideal. It's lightweight enough to drape elegantly over your shoulders but warm enough for cooler evening temperatures. It also fits perfectly in your clutch if you don't need it!", "timestamp": generate_timestamp(base_date, 3)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 3)
    })

    # ===== USER 002: Michael Rodriguez - Job Interview =====
    # Conversation 1: Interview outfit planning
    base_date = datetime.now() - timedelta(days=12, hours=19)
    conversations.append({
        "user_id": "user_002",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "I have a big interview coming up at a tech company for a Senior PM role. Need to look sharp but not overdressed.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Congratulations on the interview! Tech companies typically lean business casual to smart professional. For a Senior PM role, you want to show leadership presence without being too corporate. What's the company culture like - startup or established?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "It's a mid-size tech company, pretty modern culture. I was thinking maybe no tie?", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "No tie is a smart call for tech. I'd recommend a well-fitted navy blazer, crisp white or light blue dress shirt, and tailored chinos or dress pants in charcoal. This shows you're polished but understand the industry. What's your budget?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "I can spend up to $500. I already have nice dress shoes.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Perfect! Here's what I'd suggest: A modern-fit navy blazer ($245) that you'll wear constantly, a premium light blue dress shirt ($89), and slim-fit charcoal wool trousers ($165). Total: $499. All pieces are versatile for future interviews and work. Interested?", "timestamp": generate_timestamp(base_date, 5)},
            {"role": "user", "content": "That sounds great. What about the shirt collar - should I go spread or point?", "timestamp": generate_timestamp(base_date, 6)},
            {"role": "assistant", "content": "For a no-tie look, I'd recommend a spread collar - it lies flat and looks intentional without a tie. A point collar can sometimes look like you forgot your tie. The spread collar also frames your face well and looks modern.", "timestamp": generate_timestamp(base_date, 7)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 7)
    })

    # Conversation 2: Pre-interview check
    base_date = datetime.now() - timedelta(days=5, hours=20)
    conversations.append({
        "user_id": "user_002",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Got everything! Quick question - brown belt and shoes, or black?", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "With navy blazer and charcoal pants, you can go either way, but I'd lean towards dark brown or cognac. It's slightly less formal than black and signals that you understand modern professional style. Just make sure belt and shoes match!", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Good call. What about a watch? I have a nice silver one and a sporty black one.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Go with the silver watch - it reads as more professional and pairs well with any outfit. Keep it classic. The sporty one might send the wrong message for an interview, even at a tech company. You want subtle sophistication.", "timestamp": generate_timestamp(base_date, 3)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 3)
    })

    # ===== USER 003: Emma Thompson - Date Night =====
    # Conversation 1: Anniversary dinner outfit
    base_date = datetime.now() - timedelta(days=8, hours=14)
    conversations.append({
        "user_id": "user_003",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "It's my 2 year anniversary coming up! We're going to a fancy restaurant. I want to look amazing 💕", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "How exciting! Happy almost-anniversary! An upscale restaurant is the perfect excuse to dress up. You mentioned liking red, black, and burgundy - any of those calling to you? And are you thinking dress or a chic separates look?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Definitely a dress! Maybe red since it's around Valentine's Day? Something that says 'wow' but isn't trying too hard.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Red is perfect for the occasion! For that 'effortlessly stunning' vibe, I'd suggest a silk slip dress or a fitted midi with an interesting neckline. Nothing too tight or too revealing - the elegance of the cut does the work. What's your budget?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Up to $200 ideally. I want something special though!", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "I found the perfect piece: a wine-red satin cowl neck midi dress ($175). It has that luxe slip dress feel with a flattering draped neckline. The color is romantic without being too 'Valentine's Day obvious.' It would look stunning on you!", "timestamp": generate_timestamp(base_date, 5)},
            {"role": "user", "content": "Ooh that sounds gorgeous! What shoes would you pair with it?", "timestamp": generate_timestamp(base_date, 6)},
            {"role": "assistant", "content": "Strappy black heels would be classic and sexy. If you want to elevate it further, gold strappy sandals would add glamour. Since you're size 6.5, I have both options available. The black are $85 and the gold are $95.", "timestamp": generate_timestamp(base_date, 7)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 7)
    })

    # Conversation 2: Casual date outfit
    base_date = datetime.now() - timedelta(days=20, hours=11)
    conversations.append({
        "user_id": "user_003",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Need outfit help! Going on a museum date this weekend, want to look cute but be comfortable walking around.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Museum dates are so fun! The key is looking put-together while being practical. I'd suggest a chic midi skirt with a tucked sweater and ankle boots, or tailored trousers with a beautiful blouse. Which direction appeals to you?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "The skirt option sounds cute! What colors were you thinking?", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "For you, I'm picturing a black pleated midi skirt with a cream or burgundy fitted turtleneck. It's artistic and romantic - perfect for a museum! Add black ankle boots and a structured bag. Very 'cool art girl' energy.", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Love that! I actually have black ankle boots already. Just need the skirt and top.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Perfect! I found a sleek black satin pleated midi ($78) and a fitted burgundy ribbed turtleneck ($45). The total is $123 and both pieces will work with tons of other outfits. Want me to add them to your cart?", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 3: Quick styling question
    base_date = datetime.now() - timedelta(days=2, hours=18)
    conversations.append({
        "user_id": "user_003",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "For the anniversary dinner - hair up or down with that cowl neck dress?", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "With a cowl neck, I'd suggest hair up or swept to one side. This shows off the beautiful neckline and keeps the focus on the dress's elegant draping. A low chignon or soft updo would be perfect!", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Great idea! And for jewelry - simple or statement?", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Keep it simple to let the dress shine. Drop earrings or small hoops, maybe a delicate bracelet. Skip the necklace entirely - the cowl neckline IS your statement piece. Less is more here!", "timestamp": generate_timestamp(base_date, 3)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 3)
    })

    # ===== USER 004: David Park - Business Trip =====
    # Conversation 1: Packing for NYC trip
    base_date = datetime.now() - timedelta(days=7, hours=21)
    conversations.append({
        "user_id": "user_004",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "I have a 3-day business trip to NYC coming up. Client meetings during the day, networking event one evening. Need versatile pieces that pack well.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "NYC business trip! The key is a capsule wardrobe that mixes and matches. How many client meetings are we talking about, and what industry are your clients in?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "3 meetings over 2 days, all finance clients. Then the networking mixer is more casual tech crowd.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Finance means polished. I'd suggest: 1 navy suit, 2 dress shirts (white and light blue), 1 pair of gray wool trousers, and for the mixer you can do the suit pants with just the shirt and nice loafers. All wrinkle-resistant fabrics. Sound good?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Perfect. I need to buy the suit actually - mine is getting old. Budget around $350 for the suit?", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "For $350, I recommend this travel-ready navy wool blend suit ($345). It has stretch for comfort on flights, resists wrinkles, and looks sharp. Modern fit that works for both boardroom and rooftop bar. It's a workhorse piece.", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 2: Last-minute addition
    base_date = datetime.now() - timedelta(days=4, hours=12)
    conversations.append({
        "user_id": "user_004",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Just found out there's a dinner with executives after one of the meetings. Need something slightly more formal than my regular look.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "For executive dinner, you can elevate your existing pieces. Add a tie (navy with subtle pattern), pocket square, and make sure you have polished dress shoes. Do you have these already?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "I have dress shoes but no tie that matches. Can you find something that works with the navy suit and white shirt?", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "I found a sophisticated navy and burgundy micro-dot tie ($65) that's perfect for finance executives - classic but not boring. Pair with a white pocket square with navy edge ($25) for a polished finish. Together they'll elevate your whole look.", "timestamp": generate_timestamp(base_date, 3)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 3)
    })

    # ===== USER 005: Priya Sharma - Vacation =====
    # Conversation 1: Resort wear planning
    base_date = datetime.now() - timedelta(days=25, hours=16)
    conversations.append({
        "user_id": "user_005",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "I'm going to Bali in April! 10 days at a beach resort. I need basically a whole vacation wardrobe 🏝️", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Bali! How exciting! Let's build your resort capsule. You'll need: swimwear, coverups, flowy day dresses, comfortable walking outfits, and 2-3 dinner looks. What's your overall budget for the trip wardrobe?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Maybe $400-500 total? I have some basics but need the fun resort stuff.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Perfect! Here's my recommended breakdown:\n- 2 swimsuits ($80)\n- 2 coverups/kaftans ($70)\n- 3 day dresses ($120)\n- 2 dinner outfits ($130)\n- 1 pair dressy sandals ($50)\n\nTotal: $450. Does this split work for you?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Yes! Let's start with swimsuits and coverups. I love turquoise and coral like in my preferences.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "I found gorgeous options for you! A turquoise one-piece with a flattering wrap detail ($45) and a coral bikini with gold hardware ($35). For coverups: a white embroidered kaftan ($38) and a coral sarong wrap ($32). All very bohemian and perfect for Bali vibes!", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 2: Dinner outfits
    base_date = datetime.now() - timedelta(days=18, hours=14)
    conversations.append({
        "user_id": "user_005",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Okay now I need those dinner outfits! The resort has a nice restaurant and we want to do a fancy sunset dinner at a cliff restaurant.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Ooh, cliff restaurant sunset dinner sounds magical! For that I'd go with something flowy and romantic - think maxi dress that catches the breeze. For the resort restaurant, something more relaxed but still elevated. Thoughts on prints vs solids?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "I love prints! Something tropical or bohemian would be perfect.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Found your sunset dinner dress: a stunning coral and gold tropical print maxi with a dramatic back detail ($75). It'll photograph beautifully! For resort dinners: a turquoise paisley midi wrap dress ($55). Both pack small and don't wrinkle. Total: $130, right on budget!", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "The coral maxi sounds dreamy! What shoes for the cliff dinner?", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Gold strappy flat sandals would be perfect - dressy enough for dinner but practical for uneven cliff terrain. I have these gorgeous gold metallic gladiator sandals ($48) that would tie everything together beautifully.", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # ===== USER 006: Tyler Washington - Graduation =====
    # Conversation 1: Graduation outfit
    base_date = datetime.now() - timedelta(days=30, hours=11)
    conversations.append({
        "user_id": "user_006",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "MBA graduation in May! Need an outfit that works under the gown but also for family dinner after.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Congratulations on finishing your MBA! That's a huge accomplishment. For dual-purpose, you'll want something that looks great in photos peeking out from the gown AND stands on its own. Are you thinking full suit or something less formal?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Maybe not a full suit? It'll be warm in May. But still polished for the family photos.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Smart thinking! I'd suggest light wool dress pants, a crisp dress shirt (maybe light blue for photos - it pops against black gowns), and a blazer you can add for dinner. You could also do khakis with a navy blazer for a more preppy look. Which direction?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "The light blue shirt with dress pants sounds good. What color pants?", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Navy dress pants would be classic and photograph well. Here's my pick: lightweight navy wool trousers ($85) and a premium light blue oxford shirt ($55). Add a coordinating navy blazer ($120) for dinner. Total: $260, well under your $150 x 2 occasions mindset!", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 2: Shoes and accessories
    base_date = datetime.now() - timedelta(days=22, hours=19)
    conversations.append({
        "user_id": "user_006",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Need shoes for graduation now. Something comfortable since I'll be on my feet a lot but still dressy.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "For all-day comfort with a polished look, I'd recommend leather loafers or comfortable oxford shoes. Brown would pair perfectly with navy pants. Anything you're leaning towards?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Loafers sound good. Something I can wear to my new job too.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Great investment piece! I found cognac leather penny loafers ($95) with cushioned insoles - dressy enough for ceremonies but comfortable for long days. They'll transition perfectly to business casual office wear. They're a versatile wardrobe staple.", "timestamp": generate_timestamp(base_date, 3)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 3)
    })

    # ===== USER 007: Jordan Rivera - Fitness Event =====
    # Conversation 1: Race day outfit
    base_date = datetime.now() - timedelta(days=15, hours=8)
    conversations.append({
        "user_id": "user_007",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Half marathon coming up! Need new race day gear. My current stuff is worn out.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Exciting! A half marathon deserves fresh gear. What's the weather usually like for this race? And are you looking for top, bottom, or full outfit?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Early March so probably cool in the morning, warmer by the end. I need everything - top, shorts, and sports bra.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "For variable temps, I'd suggest moisture-wicking layers. A supportive high-impact sports bra, breathable tank that you can layer, and lightweight shorts with built-in brief. What's your budget for race gear?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Around $100 if possible. I love black and neon colors!", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Perfect color combo for visibility! Here's my race day pick: black high-support sports bra ($38), neon green mesh-panel tank ($32), and black running shorts with reflective details ($28). Total: $98. All technical fabrics that wick sweat and dry fast!", "timestamp": generate_timestamp(base_date, 5)},
            {"role": "user", "content": "Love it! What about a light layer for the start when it's cold?", "timestamp": generate_timestamp(base_date, 6)},
            {"role": "assistant", "content": "A lightweight running jacket you can tie around your waist or a throwaway long sleeve you don't mind ditching. I have a great black thumbhole long sleeve ($25) that's thin enough to not overheat but keeps the chill off at the start.", "timestamp": generate_timestamp(base_date, 7)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 7)
    })

    # Conversation 2: Post-race outfit
    base_date = datetime.now() - timedelta(days=9, hours=17)
    conversations.append({
        "user_id": "user_007",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Also need something cute for after the race - meeting friends for brunch to celebrate!", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Post-race brunch calls for comfy-cute! You'll want something easy to change into, comfortable for potentially sore muscles, but still looks put together. Thinking elevated athleisure?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Yes! Athleisure brunch vibes. Nothing too fitted though, I'll probably be bloated from all the water and gels 😂", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Ha! Totally get it. How about high-waisted joggers (forgiving waistband!), a cropped sweatshirt, and clean sneakers? I have this coral cropped hoodie ($45) that would look cute with black joggers ($55). Celebratory but comfortable!", "timestamp": generate_timestamp(base_date, 3)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 3)
    })

    # ===== USER 008: Alex Kim - Art Exhibition =====
    # Conversation 1: Gallery opening outfit
    base_date = datetime.now() - timedelta(days=11, hours=20)
    conversations.append({
        "user_id": "user_008",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "My work is being featured at a gallery opening! I need to look like an artist but also someone people want to buy art from, you know?", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Congratulations on the show! That's amazing! You want to be memorable - the outfit becomes part of your artistic brand. Creative black tie gives you room to play. Are you thinking architectural, avant-garde, or artsy-minimal?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Architectural! I want something with interesting shapes or structure. Not just a basic black dress.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Love that direction! I'm thinking: asymmetrical silhouettes, unexpected draping, maybe deconstructed tailoring. Could do a dramatic wide-leg trouser with sculptural top, or an architectural dress with interesting seaming. Budget?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Up to $300. This is a big night for me!", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Found something perfect: a black asymmetric draped dress with an origami-inspired neckline ($265). It's wearable art - exactly the statement a featured artist should make. The silhouette is dramatic but the color keeps it sophisticated. One statement piece, no accessories needed.", "timestamp": generate_timestamp(base_date, 5)},
            {"role": "user", "content": "That sounds incredible! What shoes with something that dramatic?", "timestamp": generate_timestamp(base_date, 6)},
            {"role": "assistant", "content": "Keep shoes simple to let the dress command attention. Sleek black pointed-toe mules or minimalist black heels. I have these architectural block heel mules ($85) that echo the geometric vibe without competing. Very gallery-appropriate.", "timestamp": generate_timestamp(base_date, 7)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 7)
    })

    # Conversation 2: Backup option
    base_date = datetime.now() - timedelta(days=6, hours=15)
    conversations.append({
        "user_id": "user_008",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Having second thoughts - what if the dress is too much? Do you have a backup option that's still interesting but more wearable?", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Totally valid to have options! An alternative: wide-leg black trousers with a white architectural blouse - still makes a statement but more versatile. You could even rewear pieces separately. Want to see this direction?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Yes, show me! I like having a plan B.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Here's Plan B: structured wide-leg wool trousers ($145) with a white poplin blouse featuring dramatic balloon sleeves and asymmetric buttons ($95). Total: $240. It's artistic, gallery-worthy, but each piece works in your regular wardrobe too. Very COS/Acne Studios vibe.", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "I actually love this too! Maybe I'll get both and decide day-of.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "That's the best approach for a big night! Try both with your shoes and see how you feel. Either way, you'll look like the talented artist you are. The trousers will definitely get wear beyond this event too.", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # ===== USER 009: Jennifer Martinez - School Event =====
    # Conversation 1: Conference and gala outfit
    base_date = datetime.now() - timedelta(days=16, hours=13)
    conversations.append({
        "user_id": "user_009",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "I have parent-teacher conferences at 4pm and a school gala fundraiser at 7pm on the same day. Need ONE outfit that works for both!", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "The ultimate mom challenge - I've got you! The key is a dress or jumpsuit that's polished but not too casual for conferences, and elegant enough for a gala. Can you do heels or do you need practical shoes?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Low heels max - I'll be running around. And nothing dry clean only, knowing my life something will get spilled on me!", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Ha! Real mom requirements. I'm thinking a sophisticated wrap dress in a washable fabric - flattering, practical, and transitions well. Navy or olive would look professional for conferences but dress up nicely with jewelry for the gala. Thoughts?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Navy sounds perfect. I'm a size 12 and budget is around $100 for the dress.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Found it! A navy ponte wrap dress ($89) with a flattering waist and midi length. The fabric is machine washable, wrinkle-resistant, and hides everything. For the gala, add statement earrings and swap your conference cardigan for a clutch. Done!", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 2: Accessories for gala
    base_date = datetime.now() - timedelta(days=10, hours=10)
    conversations.append({
        "user_id": "user_009",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Got the dress! Now need the accessories to glam it up for the gala portion. On a budget though.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "Accessories can totally transform that dress! For the gala, you'll want: statement earrings, a small clutch or dressy bag, and maybe a bracelet. What's your accessory budget?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Maybe $50 total? And something I'll actually wear again.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Absolutely doable! Gold drop earrings with a subtle sparkle ($22) will elevate any outfit. A small navy or blush clutch ($28) that fits your phone and lipstick. Total: $50. Both pieces you'll reach for constantly - date nights, holidays, future events!", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Perfect! Blush clutch would be pretty with navy.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Great choice! The blush adds a soft, feminine contrast to navy and photographs beautifully. Pro tip: keep the conference look simple with small studs, then do the dramatic earring swap in the car before the gala. Instant transformation!", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # ===== USER 010: Robert Anderson - Cruise =====
    # Conversation 1: Alaska cruise wardrobe
    base_date = datetime.now() - timedelta(days=45, hours=14)
    conversations.append({
        "user_id": "user_010",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Planning for our Alaska cruise in June. It's 2 weeks and the ship has formal nights. My wife says I need new clothes.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "An Alaska cruise sounds wonderful! Smart to plan ahead. You'll need layers for daytime excursions (it can be chilly even in June), casual resort wear for the ship, and 2-3 formal dinner outfits. What's your overall budget?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "Maybe $400-500? I have some basics but need the formal stuff especially.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Perfect budget. Let's prioritize: a versatile sport coat that works for formal nights ($150), 2 dress shirts ($80), dress trousers if needed ($75), and a quality layering piece for excursions ($100). Does this breakdown work?", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "That works. What color sport coat do you recommend? I mostly wear navy and tan.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "For maximum versatility on a cruise, I'd recommend a navy blazer - it works with khakis for semi-formal nights and dress trousers for formal nights. Navy photographs well against the Alaskan scenery too! Found a classic wool blend blazer ($145) that would be perfect.", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 2: Layering for excursions
    base_date = datetime.now() - timedelta(days=35, hours=11)
    conversations.append({
        "user_id": "user_010",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Now I need that layering piece for shore excursions. We're doing a glacier tour and whale watching.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "For glacier and whale watching, you'll want wind and water resistance plus warmth that's easy to pack. A packable down vest with a fleece, or a 3-in-1 jacket system. Do you run warm or cold usually?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "I run a bit cold actually. Layering is probably smart.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Then I'd recommend a quality fleece-lined softshell jacket ($95) - wind resistant for boat excursions, warm enough for glaciers, but not too bulky. In forest green or charcoal, it'll look good in photos and coordinate with everything. Plus it packs flat in your suitcase.", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "Forest green sounds nice. What do I wear under it?", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "Layer a long-sleeve moisture-wicking base shirt under a light fleece or sweater, then the jacket on top. I'd suggest bringing 2-3 merino wool base layers ($35 each) - they regulate temperature, don't hold odors, and pack small. Essential for cruises!", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    # Conversation 3: Anniversary party outfit
    base_date = datetime.now() - timedelta(days=28, hours=16)
    conversations.append({
        "user_id": "user_010",
        "session_id": generate_session_id(base_date),
        "messages": [
            {"role": "user", "content": "Almost forgot - our 50th anniversary party in July! Garden party at the country club. Need to look my best for this one.", "timestamp": generate_timestamp(base_date, 0)},
            {"role": "assistant", "content": "50 years - what a milestone! Congratulations to you both. Garden party at a country club calls for refined but not stuffy. Linen or lightweight wool blend, summer colors. Are you and your wife coordinating at all?", "timestamp": generate_timestamp(base_date, 1)},
            {"role": "user", "content": "She's wearing a light blue dress. Said I should complement but not match exactly.", "timestamp": generate_timestamp(base_date, 2)},
            {"role": "assistant", "content": "Perfect guidance from her! I'd suggest a tan or light khaki linen suit with a crisp white shirt - it complements light blue beautifully without matching. Maybe a subtle pocket square that picks up her blue. Very country club, very anniversary appropriate.", "timestamp": generate_timestamp(base_date, 3)},
            {"role": "user", "content": "I like that. Don't have a summer suit though.", "timestamp": generate_timestamp(base_date, 4)},
            {"role": "assistant", "content": "This is worth the investment for such a special occasion! I found a beautiful tan linen-blend suit ($195) that's lightweight, breathable for July, and classic. You'll also use it for summer events for years. With a white shirt ($45) and light blue pocket square ($15), you'll look wonderful standing next to your bride of 50 years.", "timestamp": generate_timestamp(base_date, 5)}
        ],
        "created_at": generate_timestamp(base_date, 0),
        "updated_at": generate_timestamp(base_date, 5)
    })

    return conversations


def insert_conversations(client, conversations):
    """Insert conversations into MongoDB."""
    db = client[TARGET_DATABASE]
    collection = db[COLLECTION_NAME]

    # Drop existing collection if it exists (for clean regeneration)
    collection.drop()
    print(f"Cleared existing '{COLLECTION_NAME}' collection")

    # Insert all conversations
    result = collection.insert_many(conversations)
    print(f"Inserted {len(result.inserted_ids)} conversations into '{COLLECTION_NAME}'")

    return result


def display_conversation_summary(conversations):
    """Display summary of generated conversations."""
    print("\n" + "=" * 60)
    print("GENERATED CONVERSATIONS SUMMARY")
    print("=" * 60)

    # Group by user
    user_convos = {}
    for conv in conversations:
        user_id = conv["user_id"]
        if user_id not in user_convos:
            user_convos[user_id] = []
        user_convos[user_id].append(conv)

    for user_id in sorted(user_convos.keys()):
        convos = user_convos[user_id]
        print(f"\n{user_id}: {len(convos)} conversation(s)")
        for conv in convos:
            msg_count = len(conv["messages"])
            print(f"  - {conv['session_id']}: {msg_count} messages")


if __name__ == "__main__":
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB Atlas...")
        client = connect_to_mongodb()

        # Test the connection
        client.admin.command("ping")
        print("Successfully connected to MongoDB Atlas!")

        # Generate conversations
        print("\nGenerating conversation histories...")
        conversations = generate_conversations()

        # Display summary
        display_conversation_summary(conversations)

        # Insert into MongoDB
        print("\n" + "=" * 60)
        print("INSERTING INTO MONGODB")
        print("=" * 60)
        insert_conversations(client, conversations)

        # Close the connection
        client.close()
        print("\nConnection closed.")

    except Exception as e:
        print(f"Error: {e}")
