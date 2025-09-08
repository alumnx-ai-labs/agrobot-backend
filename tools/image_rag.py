import os
import logging
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Optional
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dataclasses import dataclass

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

class ImageRAGTool:
    """
    ImageRAG tool for finding similar disease images using MongoDB vector search
    Filters by crop type and uses image embeddings for similarity matching
    """
    
    def __init__(self):
        """Initialize the ImageRAG tool"""
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.image_embedding_model = os.getenv("IMAGE_EMBEDDING_MODEL", "placeholder_image_model")
        
        # MongoDB connection
        self.client = None
        self.db = None
        self.collection = None
        
        # Image processing parameters
        self.input_size = (224, 224)
        self.top_k = 10  # Default number of similar images to retrieve
        self.similarity_threshold = 0.75  # Minimum similarity score
        
        # Placeholder for embedding model (to be implemented when model is decided)
        self.embedding_model = None
        
        logger.info("ImageRAGTool initialized")
    
    async def _ensure_connection(self):
        """Ensure MongoDB connection is established"""
        if self.client is None:
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client.get_default_database()
            self.collection = self.db.imagerag
            logger.info("ImageRAG: MongoDB connection established")
    
    def _preprocess_image(self, image_data: str) -> Optional[np.ndarray]:
        """
        Preprocess base64 image data for embedding generation
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Preprocessed image array or None if processing fails
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to standard size
            image = image.resize(self.input_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image)
            image_array = image_array.astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"ImageRAG: Image preprocessing failed: {str(e)}")
            return None
    
    async def _generate_image_embedding(self, image_data: str) -> Optional[List[float]]:
        """
        Generate embedding for the input image
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Image embedding vector or None if generation fails
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            if processed_image is None:
                return None
            
            # TODO: Implement actual embedding generation when model is decided
            # This is a placeholder implementation
            if self.embedding_model is None:
                logger.warning("ImageRAG: No embedding model configured, using placeholder")
                # Return a dummy embedding for testing purposes
                return [0.1] * 512  # Placeholder 512-dimensional embedding
            
            # Placeholder for actual embedding generation
            # embedding = self.embedding_model.predict(processed_image)
            # return embedding.tolist()
            
            # For now, return placeholder
            return [0.1] * 512
            
        except Exception as e:
            logger.error(f"ImageRAG: Embedding generation failed: {str(e)}")
            return None
    
    async def _vector_search(self, query_embedding: List[float], crop_type: str, top_k: int = None) -> List[dict]:
        """
        Perform vector similarity search in MongoDB
        
        Args:
            query_embedding: Query image embedding
            crop_type: Crop type for filtering
            top_k: Number of results to return
            
        Returns:
            List of similar images with metadata
        """
        try:
            if top_k is None:
                top_k = self.top_k
            
            # MongoDB vector search pipeline with crop type filtering
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "image_vector_index",  # Assumes vector index is created
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": top_k * 2  # Get more candidates for filtering
                    }
                },
                {
                    "$match": {
                        "cropType": crop_type
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "similarity_score": {"$gte": self.similarity_threshold}
                    }
                },
                {
                    "$limit": top_k
                },
                {
                    "$project": {
                        "embedding": 0  # Exclude embedding from results to save bandwidth
                    }
                }
            ]
            
            # Execute aggregation pipeline
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            
            logger.info(f"ImageRAG: Vector search returned {len(results)} similar images for crop type: {crop_type}")
            return results
            
        except Exception as e:
            logger.error(f"ImageRAG: Vector search failed: {str(e)}")
            # Fallback to text-based search if vector search fails
            return await self._fallback_text_search(crop_type, top_k)
    
    async def _fallback_text_search(self, crop_type: str, top_k: int) -> List[dict]:
        """
        Fallback text-based search when vector search fails
        
        Args:
            crop_type: Crop type for filtering
            top_k: Number of results to return
            
        Returns:
            List of documents matching crop type
        """
        try:
            logger.info("ImageRAG: Using fallback text-based search")
            
            cursor = self.collection.find(
                {"cropType": crop_type},
                {"embedding": 0}  # Exclude embedding
            ).limit(top_k)
            
            results = await cursor.to_list(length=top_k)
            
            # Add dummy similarity scores for consistency
            for result in results:
                result["similarity_score"] = 0.5  # Neutral score for fallback
            
            logger.info(f"ImageRAG: Fallback search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"ImageRAG: Fallback search failed: {str(e)}")
            return []
    
    async def query(self, image_data: str, crop_type: str, top_k: int = None) -> List[ImageRAGResult]:
        """
        Query the ImageRAG system for similar disease images
        
        Args:
            image_data: Base64 encoded image data
            crop_type: Crop type for filtering
            top_k: Number of similar images to return
            
        Returns:
            List of ImageRAGResult objects
        """
        try:
            await self._ensure_connection()
            
            if top_k is None:
                top_k = self.top_k
            
            logger.info(f"ImageRAG: Querying for similar images, crop type: {crop_type}, top_k: {top_k}")
            
            # Generate embedding for query image
            query_embedding = await self._generate_image_embedding(image_data)
            if query_embedding is None:
                logger.error("ImageRAG: Failed to generate query embedding")
                return []
            
            # Perform vector search
            similar_images = await self._vector_search(query_embedding, crop_type, top_k)
            
            if not similar_images:
                logger.warning(f"ImageRAG: No similar images found for crop type: {crop_type}")
                return []
            
            # Convert to ImageRAGResult objects
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
    
    async def get_crop_statistics(self, crop_type: str) -> dict:
        """
        Get statistics about available images for a crop type
        
        Args:
            crop_type: Crop type to analyze
            
        Returns:
            Dictionary with statistics
        """
        try:
            await self._ensure_connection()
            
            # Count total documents for this crop type
            total_count = await self.collection.count_documents({"cropType": crop_type})
            
            # Get disease class distribution
            pipeline = [
                {"$match": {"cropType": crop_type}},
                {"$group": {
                    "_id": "$class",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            disease_distribution = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Get SME distribution
            sme_pipeline = [
                {"$match": {"cropType": crop_type}},
                {"$group": {
                    "_id": "$smeRelated",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            sme_distribution = await self.collection.aggregate(sme_pipeline).to_list(length=None)
            
            return {
                "crop_type": crop_type,
                "total_images": total_count,
                "disease_distribution": {item["_id"]: item["count"] for item in disease_distribution},
                "sme_distribution": {item["_id"]: item["count"] for item in sme_distribution}
            }
            
        except Exception as e:
            logger.error(f"ImageRAG: Failed to get statistics: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> dict:
        """
        Check the health of ImageRAG system
        
        Returns:
            Dictionary with health status
        """
        try:
            await self._ensure_connection()
            
            # Test MongoDB connection
            await self.db.command("ping")
            
            # Count total documents
            total_docs = await self.collection.count_documents({})
            
            # Check if vector index exists (this would need to be implemented based on MongoDB setup)
            indexes = await self.collection.list_indexes().to_list(length=None)
            has_vector_index = any("image_vector_index" in str(idx) for idx in indexes)
            
            return {
                "mongodb_connected": True,
                "total_documents": total_docs,
                "vector_index_exists": has_vector_index,
                "embedding_model_configured": self.embedding_model is not None,
                "status": "healthy" if total_docs > 0 else "no_data"
            }
            
        except Exception as e:
            logger.error(f"ImageRAG: Health check failed: {str(e)}")
            return {
                "mongodb_connected": False,
                "error": str(e),
                "status": "unhealthy"
            }
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("ImageRAG: MongoDB connection closed")
    
    def set_embedding_model(self, model):
        """
        Set the image embedding model (to be called when model is decided)
        
        Args:
            model: The embedding model instance
        """
        self.embedding_model = model
        logger.info("ImageRAG: Embedding model configured")
    
    def get_supported_crops(self) -> List[str]:
        """
        Get list of supported crop types (would be implemented based on actual data)
        
        Returns:
            List of supported crop types
        """
        # This would typically query the database for unique crop types
        # For now, return common crops as placeholder
        return [
            "tomato", "wheat", "rice", "corn", "potato", 
            "cotton", "soybean", "apple", "grape", "citrus"
        ]