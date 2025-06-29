import json

with open("calorie_data/calorie_lookup.json") as f:
    calorie_map = json.load(f)

def estimate_calories(food_label):
    label = food_label.lower().replace(" ", "_")
    return calorie_map.get(label, "Unknown")
