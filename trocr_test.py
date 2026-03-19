from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance
import sys

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

print("🔧 Loading model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

print(f"📸 Preprocessing: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("L")          # grayscale
image = ImageOps.invert(image)                       # dark text on white bg
image = ImageEnhance.Contrast(image).enhance(2.0)    # boost contrast
image = image.convert("RGB")

print("🧠 Running OCR...")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n✅ Transcription:")
print("────────────────────────")
print(text)
print("────────────────────────")
