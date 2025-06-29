import json
import os

# Load calorie lookup table
lookup_path = os.path.join("calorie_data", "calorie_lookup.json")

with open(lookup_path, "r") as f:
    calorie_map = json.load(f)

def estimate_calories(food_label):
    key = food_label.lower().replace(" ", "_")
    return calorie_map.get(key, "Unknown")
