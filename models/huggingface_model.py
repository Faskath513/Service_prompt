import os
from dotenv import load_dotenv
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load environment variables from .env file
load_dotenv()

# Initialize the Hugging Face GPT-2 pipeline
generator = pipeline("text-generation", model="distilgpt2" ,)
# Load the DistilBERT model for service classification
classification_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
classification_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Define possible service types for classification (you can expand this list)
service_types = {
    "HVAC": [
        "HVAC", "heating", "ventilation", "air conditioning", "cooling", "maintenance",
        "temperature control", "furnace", "boiler", "duct cleaning", "thermostat"
    ],
    "Plumber": [
        "plumber", "pipe", "leak", "drain", "water", "faucet", "toilet", "sewer", "clog",
        "plumbing", "overflow", "shower", "sump pump", "drainage", "water heater", "fix leaking pipe", "clear drain clog", "install faucet", "repair toilet",
        "water heater repair", "fix shower drain", "leak detection", "pipe inspection"
    ],
    "Electrician": [
        "electrician", "wiring", "circuit", "lights", "electric", "socket", "outlet", "breaker",
        "voltage", "current", "fuse", "panel", "generator", "switch", "electrical repair", "install light fixtures", "replace socket", "wiring installation", "electrical troubleshooting",
        "fuse replacement", "outlet repair", "install ceiling fan", "electric meter reading"
    ],
    "Cleaner": [
        "cleaning", "housekeeping", "vacuum", "clean", "dusting", "laundry", "mopping",
        "scrubbing", "janitorial", "deep clean", "organization", "decluttering", "sanitizing","house cleaning", "office cleaning", "deep cleaning", "move-out cleaning", "vacuuming",
        "dusting", "mopping", "window cleaning", "carpet cleaning", "laundry"
    ],
    "Event Planning Services": [
        "party planning", "wedding coordination", "event setup", "catering service",
        "venue selection", "guest list management", "decorations", "birthday party planning"
    ],
    "Gardener": [
        "gardener", "landscaping", "lawn", "plant", "tree", "mowing", "pruning", "hedge",
        "flower bed", "gardening", "weeding", "mulching", "fertilizer", "irrigation", "compost","lawn mowing", "weed control", "pruning", "planting flowers", "garden cleaning",
        "hedge trimming", "landscaping consultation", "mulching", "tree trimming"
    ],
    "Painter": [
        "painting", "wall", "decorate", "brush", "color", "coat", "spray", "painter",
        "paint job", "roller", "stain", "artistic", "wall design", "fence painting","room painting", "wall touch-up", "fence painting", "exterior house painting",
        "ceiling painting", "painting furniture", "door painting"
    ],
    "Carpenter": [
        "carpenter", "wood", "furniture", "cabinet", "cutting", "frame", "construction",
        "repair wood", "joinery", "decking", "woodwork", "sanding", "doors", "shelves","fix broken furniture", "install wooden shelves", "build small furniture", "repair wooden doors",
        "assemble bookshelf", "repair cabinets", "install wooden flooring"
    ],
    "Mover": [
        "moving", "movers", "relocation", "transport", "packing", "delivery", "unloading",
        "storage", "house move", "office relocation", "furniture moving", "loading", "pack and move", "furniture moving", "loading and unloading", "office relocation",
        "small house move", "deliver items", "local moving"
    ],
    "Roofing": [
        "roofing", "roof", "shingles", "tiles", "leak", "construction",
        "waterproofing", "installation", "gutters", "skylights", "re-roofing", "insulation"
    ],
    "Pest Control": [
        "pest control", "extermination", "rodents", "insects", "bugs", "fumigation",
        "termite", "ant", "cockroach", "mice", "rat", "mosquito", "spider", "wasp", "pests", "termite inspection", "ant removal", "cockroach extermination", "mosquito control",
        "rodent control", "bedbug treatment", "general pest removal", "spider control"
    ],
    "Locksmith": [
        "locksmith", "lock", "key", "security", "rekey", "unlock", "repair lock", "installation",
        "safe", "key duplication", "smart lock", "deadbolt", "door lock", "access control","lock rekeying", "lock repair", "install new lock", "unlock door",
        "key duplication", "safe opening", "security system installation"
    ],
    "Masonry": [
        "masonry", "stonework", "bricks", "mortar", "concrete", "foundation", "chimney",
        "paving", "patio", "wall", "stone", "blockwork", "cement", "driveway", "bricklaying"
    ],
    "IT Support": [
        "IT support", "technical support", "computer", "network", "system", "hardware",
        "software", "server", "IT troubleshooting", "cybersecurity", "data recovery", "setup","computer setup", "network troubleshooting", "virus removal", "data recovery",
        "Wi-Fi setup", "software installation", "email setup", "PC repair","laptop"
    ],
    "House Renovation": [
        "house renovation", "remodeling", "construction", "building", "remodel",
        "home improvement", "interior design", "extension", "kitchen renovation",
        "bathroom remodel", "flooring", "tiling", "wall remodeling"
    ],
    "House Cleaning": [
        "general house cleaning", "bathroom cleaning", "kitchen cleaning", "floor cleaning",
        "dusting", "organizing", "decluttering", "window washing"
    ],
    "Carpet Cleaning": [
        "steam cleaning", "dry cleaning", "stain removal", "deep cleaning", "shampooing"
    ],
    "Swimming Pool": [
        "swimming pool", "pool maintenance", "cleaning pool", "pool repair",
        "swimming pool installation", "chlorine", "filter system", "pool heater",
        "pool design", "water treatment", "pool resurfacing"
    ],
    "Window Repair Services": [
        "replace broken glass", "fix window lock", "clean windows", "seal window leaks",
        "window tinting", "install blinds or curtains"
    ],
    "Window Cleaning": [
        "window cleaning", "glass cleaning", "windows", "exterior cleaning",
        "interior window", "pane cleaning", "streak-free", "squeegee", "residential windows"
    ],
    "Security Services": [
        "security", "surveillance", "cameras", "alarm system", "monitoring", "security guard",
        "safety", "motion sensors", "access control", "home security", "security patrol"
    ],
    "Transportation": [
        "transportation", "vehicle", "delivery", "freight", "logistics", "shipping", "hauling",
        "rideshare", "moving goods", "courier service", "transport vehicles"
    ],
    "Event Planning": [
        "event planning", "event coordinator", "wedding planning", "party planning",
        "celebration", "organizer", "event setup", "catering coordination", "venue management"
    ],
    "Massage": [
        "massage", "therapy", "relaxation", "massage therapy", "swedish massage",
        "deep tissue", "hot stone massage", "therapeutic massage", "spa treatment", "relaxation massage", "deep tissue massage", "hot stone therapy", "sports massage"
    ],
    "Mechanic": [
        "mechanic", "auto repair", "vehicle maintenance", "car repair", "engine repair",
        "brakes", "transmission", "tune-up", "tires", "battery replacement", "alignment",
        "oil change", "suspension", "diagnostics", "exhaust repair","oil change", "battery replacement", "brake inspection", "tire change",
        "car diagnostics", "engine repair", "car washing", "clutch replacement"
    ],
    "Personal Chef": [
        "personal chef", "private chef", "meal prep", "cooking service", "meal delivery",
        "catering", "specialty meals", "food preparation", "dietary restrictions", "gourmet chef"
    ],
    "Tutor": [
        "tutor", "teaching", "education", "math tutor", "english tutor", "online classes",
        "homework help", "science tutor", "test prep", "personalized teaching","math tutoring", "English tutoring", "test prep", "homework help", "study assistance",
        "language tutoring", "science tutoring", "online classes"
    ],
    "Beauty Salon": [
        "beauty salon", "haircut", "stylist", "coloring", "hair", "manicure",
        "pedicure", "facial", "waxing", "beauty treatment", "makeup", "spa"
    ],
    "Auto Mechanic": [
        "auto mechanic", "car repair", "engine", "brakes", "oil change", "tires",
        "diagnostic", "vehicle maintenance", "auto service", "alignment", "tuning", "battery"
    ],
    "Car Wash Services": [
        "car interior cleaning", "exterior car wash", "waxing", "detailing", "polishing",
        "tire cleaning", "engine cleaning"
    ],
    "Handyman": [
        "handyman", "home repair", "general maintenance", "fixing", "odd jobs",
        "small repairs", "door fixing", "painting walls", "light installation","assemble furniture", "fix door lock", "repair small appliances", "patch drywall",
        "install shelves", "minor home repairs", "furniture assembly", "mount TV", "install curtain rods"
    ],
    "Catering": [
        "catering", "food service", "event food", "meals", "wedding catering",
        "party catering", "buffet", "snacks", "cocktail party", "plated meals"
    ],
    "Dog Walker": [
        "dog walking", "pet care", "dog walker", "walking dog", "pet walking",
        "dog exercise", "dog sitting", "pet sitter", "dog playtime"
    ],
    "Child Care": [
        "child care", "babysitting", "nanny", "daycare", "children supervision",
        "kids care", "baby care", "after-school care", "child support", "nursery"
    ],
    "Elder Care": [
        "elder care", "senior care", "nursing", "home care", "elderly assistance",
        "caregiving", "companion care", "medication support", "health monitoring"
    ],
    "Fitness Trainer": [
        "fitness trainer", "personal trainer", "gym coach", "workout", "exercise",
        "fitness goals", "strength training", "cardio sessions", "diet planning"
    ],
    "Photography": [
        "photography", "photo shoot", "event photographer", "wedding photography",
        "portrait photographer", "photo editing", "product photography", "headshots","portrait photography", "event photography", "product photography",
        "headshots", "family photoshoot", "wedding photography", "photo editing"
    ],
    "Web Development": [
        "web development", "website creation", "frontend development", "backend development",
        "web design", "coding", "HTML", "CSS", "JavaScript", "ecommerce website"
    ],
    "Bike Repair Services": [
        "fix flat tire", "chain replacement", "brake repair", "bike tune-up", "gear adjustment"
    ],
    "Legal Services": [
        "legal services", "lawyer", "attorney", "legal advice", "contract drafting",
        "notary", "litigation", "legal consultation", "court cases"
    ],
    "Translation": [
        "translation", "language translation", "translator", "document translation",
        "interpretation", "localization", "multilingual", "text translation"
    ],
    "Pet Grooming": [
        "pet grooming", "dog grooming", "cat grooming", "pet bathing", "pet haircut",
        "nail clipping", "fur trimming", "pet styling", "pet spa"
    ],
    "Architect": [
        "architect", "building design", "blueprints", "construction plans",
        "interior design", "urban planning", "space planning", "renovation", "architecture"
    ],
    "Event Photography": [
        "event photography", "event photographer", "wedding photographer",
        "party photographer", "conference photographer", "corporate event photography",
        "event photos", "photojournalism", "event videography"
    ],
    "Home Inspection": [
        "home inspection", "property inspection", "real estate inspection",
        "building inspection", "house assessment", "pre-purchase inspection", "termite inspection"
    ],
    "Virtual Assistant": [
        "virtual assistant", "admin assistant", "personal assistant", "task management",
        "schedule management", "customer service", "virtual office support"
    ],
    "Cleaning Services": [
        "cleaning service", "house cleaning", "office cleaning", "commercial cleaning",
        "deep cleaning", "move-in cleaning", "move-out cleaning", "sanitization"
    ],
    "Digital Marketing": [
        "digital marketing", "SEO", "content marketing", "PPC", "social media marketing",
        "online advertising", "email marketing", "affiliate marketing", "branding", "search engine optimization"
    ],
    "Real Estate Agent": [
        "real estate agent", "real estate broker", "property agent", "home buying",
        "home selling", "real estate listing", "property management", "real estate investment"
    ],
    "Financial Advisor": [
        "financial advisor", "investment advisor", "retirement planning", "tax advisor",
        "financial planning", "wealth management", "investment strategies", "budgeting", "insurance"
    ],
    "Life Coach": [
        "life coach", "personal coaching", "career coach", "motivation",
        "goal setting", "personal growth", "self-improvement", "coaching sessions"
    ],
    "Dietitian": [
        "dietitian", "nutritionist", "diet planning", "meal planning",
        "weight loss", "healthy eating", "specialized diets", "nutrition counseling"
    ],
    "Home Security": [
        "home security", "security system", "alarm installation", "surveillance cameras",
        "home surveillance", "security monitoring", "access control", "smart home security"
    ],
    "3D Printing": [
        "3D printing", "3D design", "rapid prototyping", "additive manufacturing",
        "3D models", "CAD design", "3D printed objects", "custom 3D printing"
    ],
    "Tattoo Artist": [
        "tattoo artist", "tattoo design", "custom tattoos", "ink", "tattoo studio",
        "body art", "piercing", "tattoo consultation", "skin art"
    ],
    "Voiceover Artist": [
        "voiceover artist", "voice acting", "audio recording", "narration",
        "voice talent", "voice recordings", "commercial voiceover", "character voices"
    ],
    "Courier": [
        "courier", "delivery service", "express delivery", "shipping",
        "local delivery", "overnight shipping", "parcel delivery", "package courier"
    ],
    "Handmade Crafts": [
        "handmade crafts", "custom crafts", "artisanal products", "handmade jewelry",
        "handmade gifts", "crafting services", "personalized crafts", "handcrafted items"
    ],
    "Nutritionist": [
        "nutritionist", "healthy eating", "diet plans", "nutritional counseling",
        "meal plans", "weight management", "food advice", "food allergies", "specialized diets"
    ],
    "Voice Coach": [
        "voice coach", "singing lessons", "voice training", "vocal coaching",
        "speech coaching", "public speaking", "breath control", "pitch training"
    ],
    "Baker": [
        "baker", "pastry", "cakes", "cookies", "bread", "cupcakes",
        "custom cakes", "desserts", "baking service", "baked goods"
    ],
    "Interior Designer": [
        "interior designer", "home decor", "space planning", "design consultant",
        "furniture selection", "home styling", "color schemes", "home renovations"
    ],
    "Translator (Sign Language)": [
        "sign language interpreter", "sign language translation", "ASL interpreter",
        "deaf communication", "sign language classes", "language support", "accessibility"
    ],
    "Elderly Transportation": [
        "elderly transportation", "senior transport", "medical transportation", "senior rides",
        "elderly mobility", "senior care transportation", "transportation for elderly"
    ],
    "Mobile App Development": [
        "mobile app development", "iOS app development", "Android app development", "app design",
        "app programming", "mobile app development services", "mobile UX/UI design"
    ],
    "Custom Clothing": [
        "custom clothing", "personalized clothing", "customized t-shirts", "embroidery",
        "screen printing", "custom apparel", "fashion design", "tailoring services"
    ],
    "Social Media Management": [
        "social media management", "content creation", "social media marketing", "brand management",
        "content scheduling", "community management", "social media strategy"
    ],
    "Pool Installation": [
        "pool installation", "swimming pool installation", "in-ground pool", "above-ground pool",
        "pool design", "landscape pool", "custom pool installation", "pool construction"
    ],
    "Glass Replacement Service":[" replace a broken window","broken glass"]
}
def classify_service_type(service_name: str) -> str:
    """Classifies the service type based on keywords in the service name."""
    service_name_lower = service_name.lower()

    for service_type, keywords in service_types.items():
        if any(keyword in service_name_lower for keyword in keywords):
            return service_type

    return "Unknown"  # If no match is found


def generate_service_description(service_name: str) -> dict:
    try:
        # Classify the service type based on the input
        service_type = classify_service_type(service_name)

        # Generate the description using GPT-2
        prompt = f"Provide a detailed description for a {service_name} service."
        result = generator(prompt, max_length=200, num_return_sequences=1)  # Increased max_length for better description

        # Extract and return the generated description
        description = result[0]['generated_text'].strip()

        return {
            "description": description
        }

    except Exception as e:
        print(f"Error generating description: {e}")
        return {"service_type": "Unknown", "description": "Sorry, I couldn't generate a description at the moment."}
