import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import classify_image
from src.calories import estimate_calories

image_path = "sample_image.jpg"

predictions = classify_image(image_path)
for label, prob in predictions:
    print(f"🍽️ {label}: {prob:.2%}")
    calories = estimate_calories(label)
    print(f"🔥 Estimated calories: {calories if calories != 'Unknown' else 'No estimate available'}")
