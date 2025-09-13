import os
import logging
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
import uuid
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== USER INPUTS ====================
# Please update these global variables with your specific values.
# The script will use these values for all images in the specified folder.

input_folder = "/path/to/your/image/folder"  # Path to the folder containing images
image_class = "powdery_mildew"            # The class of all images in the folder
image_source = "Farmer_Upload"            # The source of the images (e.g., 'Farmer_Upload')
crop_type = "tomato"                      # The crop type for all images
sme_related = "Dr. Jane Doe"              # The SME related to these images

# ======================================================

# Pinecone and Gemini API keys from .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_6o3mJh_Jiw2umo4cxcjq5kqNqov4NRgXV9SfkkDGbq2n3RbKiuHaztoqiDMJW1utqoivLf")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "alumnx-alumni")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# GEMINI_API_KEY is no longer needed for this specific script

if not PINECONE_API_KEY:
    raise ValueError("Missing required environment variables: PINECONE_API_KEY.")

# Custom class for CLIP embeddings
class CLIPEmbeddings:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image_path: str) -> List[float]:
        """Generates an embedding for an image from a file path."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten().tolist()

# Initialize Pinecone and the new embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = CLIPEmbeddings()

async def ensure_index_exists():
    """Ensure Pinecone index exists, create if not."""
    try:
        existing_indexes = pc.list_indexes().names
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=512,  # The dimension for CLIP ViT-B-32
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                await asyncio.sleep(1)
            logger.info(f"Index {PINECONE_INDEX_NAME} created and ready.")
        else:
            logger.info(f"Index {PINECONE_INDEX_NAME} already exists.")
    except Exception as e:
        logger.error(f"Error ensuring index exists: {e}")
        raise

async def ingest_folder_data(
    folder_path: str,
    image_class: str,
    image_source: str,
    crop_type: str,
    sme_related: str
):
    """
    Ingest all images from a folder into the Pinecone index.
    """
    await ensure_index_exists()
    index = pc.Index(PINECONE_INDEX_NAME)
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    vectors_to_upsert = []
    
    logger.info(f"Starting ingestion from folder: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            
            doc_id = str(uuid.uuid4())
            
            try:
                # Generate a real image embedding using the CLIP model
                embedding = embeddings.embed_image(file_path)
            except Exception as e:
                logger.error(f"Failed to generate embedding for {filename}: {e}")
                continue # Skip this file and continue with the next
            
            # Prepare the metadata for the image
            metadata = {
                "class": image_class,
                "imagePath": file_path,
                "source": image_source,
                "cropType": crop_type,
                "smeRelated": sme_related
            }
            
            # Prepare the vector for upsert
            vector = {
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }
            vectors_to_upsert.append(vector)
            logger.info(f"Prepared vector for {filename}")

    if not vectors_to_upsert:
        logger.warning(f"No supported image files found in {folder_path}")
        return

    logger.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
    index.upsert(vectors=vectors_to_upsert, batch_size=100) # Use batching for large folders
    logger.info("Ingestion complete.")

async def main():
    """Main function to run the folder ingestion process."""
    if not os.gh.isdir(input_folder):
        logger.error(f"The provided input folder '{input_folder}' does not exist.")
        return
    
    await ingest_folder_data(
        folder_path=input_folder,
        image_class=image_class,
        image_source=image_source,
        crop_type=crop_type,
        sme_related=sme_related
    )

if __name__ == "__main__":
    asyncio.run(main())
