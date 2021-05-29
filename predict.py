from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

image = Image.open('test.png').convert('RGB')

feature_extractor = AutoFeatureExtractor.from_pretrained("dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("dit-base-finetuned-rvlcdip")

# print("feature",feature_extractor)
inputs = feature_extractor(images=image, return_tensors="pt")
# print("inputs: " , inputs)
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 16 RVL-CDIP classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
