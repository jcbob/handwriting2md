from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import sys

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

print("🔧 Loading printed TrOCR model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

image = Image.open(IMAGE_PATH).convert("RGB")

print("🧠 Running OCR...")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n✅ Transcription:")
print(text)
