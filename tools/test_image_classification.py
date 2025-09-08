import base64
import asyncio
from image_classification import ImageClassificationTool  # <-- changed import

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

async def main():
    image_path = "test_images/Anthracnose004.jpg"  # Change this to your image file
    crop_type = "default"  # Or specify your crop type if you have crop-specific models

    image_data = encode_image_to_base64(image_path)
    classifier = ImageClassificationTool()
    result = await classifier.predict(image_data, crop_type)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())