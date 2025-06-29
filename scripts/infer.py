from src.classifier import classify_image
from src.calories import estimate_calories

image_path = "sample_image.jpg"

predicted_food = classify_image(image_path)
calories = estimate_calories(predicted_food)

print(f"🍽️ Predicted dish: {predicted_food}")
print(f"🔥 Estimated calories: {calories if calories != 'Unknown' else 'No estimate available'}")
