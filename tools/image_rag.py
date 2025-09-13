import os
import logging
import base64
import numpy as np
import asyncio
from io import BytesIO
from PIL import Image
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import torch
from transformers import CLIPProcessor, CLIPModel

from pinecone import Pinecone, ServerlessSpec
# Removed the import for GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

@dataclass
class ImageRAGResult:
    """Result structure for ImageRAG queries"""
    disease_name: str
    confidence: float
    image_url: str
    description: str
    source: str
    crop_type: str
    sme_related: str

# Custom class for CLIP embeddings, similar to the ingestion script
class CLIPEmbeddings:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image_data: str) -> List[float]:
        """Generates an embedding for an image from a base64 string."""
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten().tolist()

class ImageRAGTool:
    """
    ImageRAG tool for finding similar disease images using Pinecone vector search.
    Filters by crop type and uses image embeddings for similarity matching.
    """
    
    def __init__(self):
        """Initialize the ImageRAG tool with Pinecone and embedding components."""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "pcsk_6o3mJh_Jiw2umo4cxcjq5kqNqov4NRgXV9SfkkDGbq2n3RbKiuHaztoqiDMJW1utqoivLf")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "alumnx-alumni")
        
        if not self.pinecone_api_key:
            raise ValueError("Missing required API keys: PINECONE_API_KEY.")
        
        # Pinecone and embedding parameters
        self.top_k = 10  # Default number of similar images to retrieve
        self.similarity_threshold = 0.75  # Minimum similarity score
        self.input_size = (224, 224) # Still relevant for consistency if needed
        
        self.pc = None
        self.embeddings = None
        self.index = None
        
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all required components (Pinecone and Embeddings)."""
        try:
            # Initialize Pinecone with the new SDK
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Initialize CLIP embeddings
            self.embeddings = CLIPEmbeddings()
            
            # Connect to the Pinecone index.
            self.index = self.pc.Index(self.pinecone_index_name)

            logger.info("Pinecone client and embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _ensure_index_exists(self):
        """Ensure Pinecone index exists, create if not."""
        try:
            existing_indexes = self.pc.list_indexes().names
            if self.pinecone_index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.pinecone_index_name}")
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=512,  # Updated dimension for CLIP
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD", "aws"),
                        region=os.getenv("PINECONE_REGION", "us-east-1")
                    )
                )
                
                while not self.pc.describe_index(self.pinecone_index_name).status['ready']:
                    await asyncio.sleep(1)
                
                logger.info(f"Index {self.pinecone_index_name} created and ready.")
            else:
                logger.info(f"Index {self.pinecone_index_name} already exists.")
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise

    # The _preprocess_image and _generate_image_embedding methods are now simplified
    # to use the new CLIPEmbeddings class directly.

    async def _vector_search(self, query_embedding: List[float], crop_type: str, top_k: int = None) -> List[dict]:
        """
        Perform vector similarity search in Pinecone.
        """
        try:
            if top_k is None:
                top_k = self.top_k
            
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={
                    "cropType": {"$eq": crop_type}
                }
            )
            
            results = []
            for match in search_results.matches:
                if match.score >= self.similarity_threshold:
                    result_doc = match.metadata
                    result_doc["similarity_score"] = match.score
                    results.append(result_doc)

            logger.info(f"ImageRAG: Vector search returned {len(results)} similar images for crop type: {crop_type}")
            return results
            
        except Exception as e:
            logger.error(f"ImageRAG: Vector search failed: {str(e)}")
            return []
    
    async def query(self, image_data: str, crop_type: str, top_k: int = None) -> List[ImageRAGResult]:
        """
        Query the ImageRAG system for similar disease images.
        """
        try:
            await self._ensure_index_exists()
            
            if top_k is None:
                top_k = self.top_k
            
            logger.info(f"ImageRAG: Querying for similar images, crop type: {crop_type}, top_k: {top_k}")
            
            # Generate embedding for query image using the new CLIPEmbeddings class
            query_embedding = self.embeddings.embed_image(image_data)
            if query_embedding is None:
                logger.error("ImageRAG: Failed to generate query embedding")
                return []
            
            similar_images = await self._vector_search(query_embedding, crop_type, top_k)
            
            if not similar_images:
                logger.warning(f"ImageRAG: No similar images found for crop type: {crop_type}")
                return []
            
            results = []
            for img_doc in similar_images:
                try:
                    result = ImageRAGResult(
                        disease_name=img_doc.get("class", "unknown_disease"),
                        confidence=float(img_doc.get("similarity_score", 0.0)),
                        image_url=img_doc.get("imagePath", ""),
                        description=f"Similar disease image from {img_doc.get('source', 'unknown source')}",
                        source=img_doc.get("source", ""),
                        crop_type=img_doc.get("cropType", crop_type),
                        sme_related=img_doc.get("smeRelated", "")
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"ImageRAG: Failed to parse document: {str(e)}")
                    continue
            
            logger.info(f"ImageRAG: Successfully retrieved {len(results)} similar images")
            return results
            
        except Exception as e:
            logger.error(f"ImageRAG: Query failed: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of ImageRAG system.
        """
        status = {
            "pinecone_connection": False,
            "vector_index_exists": False,
            "embedding_model_configured": self.embeddings is not None,
            "status": "unhealthy"
        }
        
        try:
            indexes = self.pc.list_indexes().names
            status["pinecone_connection"] = True
            status["vector_index_exists"] = self.pinecone_index_name in indexes
            
            if status["vector_index_exists"]:
                stats = self.index.describe_index_stats()
                total_vectors = stats.total_vector_count
                status["total_documents"] = total_vectors
                status["status"] = "healthy" if total_vectors > 0 else "no_data"
            else:
                status["status"] = "unhealthy"
            
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            status["error"] = str(e)
            
        return status
