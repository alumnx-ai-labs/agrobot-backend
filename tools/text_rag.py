import os
import logging
from typing import List, Optional
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class TextRAGResult:
    """Result structure for TextRAG queries"""
    disease_name: str
    symptoms: str
    prevention: str
    treatment: str
    additional_info: str
    source_documents: List[str]
    confidence_score: float

class TextRAGTool:
    """
    TextRAG tool for retrieving disease information using Google embeddings and MongoDB
    Filters by SME advisor and crop type for targeted results
    """
    
    def __init__(self):
        """Initialize the TextRAG tool with Google embeddings and MongoDB connection"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.embedding_model = os.getenv("TEXT_EMBEDDING_MODEL", "models/embedding-001")
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Initialize Google AI
        genai.configure(api_key=self.google_api_key)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.google_api_key
        )
        
        # Initialize MongoDB client
        self.client = None
        self.db = None
        self.collection = None
        
        logger.info("TextRAGTool initialized")
    
    async def _ensure_connection(self):
        """Ensure MongoDB connection is established"""
        if self.client is None:
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client.get_default_database()
            self.collection = self.db.smetextrag
            logger.info("MongoDB connection established")
    
    async def query(self, disease_name: str, sme_advisor: str, crop_type: str = None) -> TextRAGResult:
        """
        Query the TextRAG system for disease information
        
        Args:
            disease_name: The disease to search for
            sme_advisor: SME advisor name for filtering
            crop_type: Crop type for filtering (optional, can be taken from workflow state)
        
        Returns:
            TextRAGResult with disease information and preventive measures
        """
        try:
            await self._ensure_connection()
            
            logger.info(f"TextRAG: Querying for disease '{disease_name}', SME: '{sme_advisor}', Crop: '{crop_type}'")
            
            # Generate embedding for the disease query
            query_embedding = await self._generate_embedding(disease_name)
            
            # Build MongoDB filter
            mongo_filter = {
                "smeAdvisorName": sme_advisor
            }
            
            if crop_type:
                mongo_filter["cropType"] = crop_type
            
            logger.info(f"TextRAG: MongoDB filter: {mongo_filter}")
            
            # Perform vector similarity search
            relevant_chunks = await self._vector_search(query_embedding, mongo_filter)
            
            if not relevant_chunks:
                logger.warning(f"TextRAG: No relevant documents found for disease '{disease_name}' with given filters")
                return self._create_empty_result(disease_name)
            
            logger.info(f"TextRAG: Found {len(relevant_chunks)} relevant chunks")
            
            # Generate comprehensive response using Google's generative model
            result = await self._generate_response(disease_name, relevant_chunks)
            
            logger.info(f"TextRAG: Successfully generated response for '{disease_name}'")
            return result
            
        except Exception as e:
            logger.error(f"TextRAG: Error querying for '{disease_name}': {str(e)}")
            return self._create_error_result(disease_name, str(e))
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the input text"""
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"TextRAG: Embedding generation failed: {str(e)}")
            raise
    
    async def _vector_search(self, query_embedding: List[float], mongo_filter: dict, top_k: int = 10) -> List[dict]:
        """
        Perform vector similarity search in MongoDB
        
        Args:
            query_embedding: Query vector
            mongo_filter: MongoDB filter criteria
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks
        """
        try:
            # MongoDB vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Assumes vector index is created
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": top_k
                    }
                },
                {
                    "$match": mongo_filter
                },
                {
                    "$project": {
                        "text": 1,
                        "pdfName": 1,
                        "cropType": 1,
                        "smeAdvisorName": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Execute aggregation pipeline
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            
            # Filter by minimum similarity threshold
            threshold = 0.7
            filtered_results = [doc for doc in results if doc.get('score', 0) >= threshold]
            
            logger.info(f"TextRAG: Vector search returned {len(results)} results, {len(filtered_results)} above threshold")
            return filtered_results
            
        except Exception as e:
            logger.error(f"TextRAG: Vector search failed: {str(e)}")
            return []
    
    async def _generate_response(self, disease_name: str, relevant_chunks: List[dict]) -> TextRAGResult:
        """
        Generate comprehensive disease information using Google's generative model
        
        Args:
            disease_name: The disease name
            relevant_chunks: Relevant document chunks from vector search
            
        Returns:
            TextRAGResult with structured disease information
        """
        try:
            # Prepare context from relevant chunks
            context = "\n\n".join([
                f"Source: {chunk.get('pdfName', 'Unknown')} (Score: {chunk.get('score', 0):.2f})\n{chunk.get('text', '')}"
                for chunk in relevant_chunks
            ])
            
            # Create prompt for structured information extraction
            prompt = f"""
Based on the following agricultural and plant pathology documents, provide comprehensive information about the disease: {disease_name}

Context from expert documents:
{context}

Please provide a structured response with the following sections:

1. SYMPTOMS: Detailed description of disease symptoms
2. PREVENTION: Preventive measures and best practices
3. TREATMENT: Treatment options and management strategies
4. ADDITIONAL_INFO: Any additional relevant information, cultural practices, or expert insights

Format your response as follows:
SYMPTOMS: [detailed symptoms]
PREVENTION: [preventive measures]
TREATMENT: [treatment options]
ADDITIONAL_INFO: [additional information]

Ensure all information is specific to {disease_name} and based on the provided expert documents.
"""
            
            # Generate response using Google's model
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            # Parse the structured response
            response_text = response.text if response else ""
            parsed_info = self._parse_structured_response(response_text)
            
            # Extract source documents
            source_documents = list(set([chunk.get('pdfName', 'Unknown') for chunk in relevant_chunks]))
            
            # Calculate confidence score based on number of relevant chunks and their scores
            avg_score = sum([chunk.get('score', 0) for chunk in relevant_chunks]) / len(relevant_chunks)
            confidence_score = min(avg_score * (len(relevant_chunks) / 10), 1.0)  # Normalize
            
            return TextRAGResult(
                disease_name=disease_name,
                symptoms=parsed_info.get('symptoms', ''),
                prevention=parsed_info.get('prevention', ''),
                treatment=parsed_info.get('treatment', ''),
                additional_info=parsed_info.get('additional_info', ''),
                source_documents=source_documents,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"TextRAG: Response generation failed: {str(e)}")
            return self._create_error_result(disease_name, f"Response generation failed: {str(e)}")
    
    def _parse_structured_response(self, response_text: str) -> dict:
        """Parse the structured response from the generative model"""
        try:
            sections = {
                'symptoms': '',
                'prevention': '',
                'treatment': '',
                'additional_info': ''
            }
            
            lines = response_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('SYMPTOMS:'):
                    current_section = 'symptoms'
                    sections[current_section] = line.replace('SYMPTOMS:', '').strip()
                elif line.startswith('PREVENTION:'):
                    current_section = 'prevention'
                    sections[current_section] = line.replace('PREVENTION:', '').strip()
                elif line.startswith('TREATMENT:'):
                    current_section = 'treatment'
                    sections[current_section] = line.replace('TREATMENT:', '').strip()
                elif line.startswith('ADDITIONAL_INFO:'):
                    current_section = 'additional_info'
                    sections[current_section] = line.replace('ADDITIONAL_INFO:', '').strip()
                elif current_section and line:
                    # Continue appending to current section
                    if sections[current_section]:
                        sections[current_section] += ' ' + line
                    else:
                        sections[current_section] = line
            
            return sections
            
        except Exception as e:
            logger.error(f"TextRAG: Response parsing failed: {str(e)}")
            return {
                'symptoms': response_text,
                'prevention': '',
                'treatment': '',
                'additional_info': ''
            }
    
    def _create_empty_result(self, disease_name: str) -> TextRAGResult:
        """Create an empty result when no documents are found"""
        return TextRAGResult(
            disease_name=disease_name,
            symptoms="No specific information available in the knowledge base.",
            prevention="General preventive measures recommended. Consult with agricultural extension services.",
            treatment="Consult with plant pathology experts for treatment recommendations.",
            additional_info="No additional information available in the current knowledge base.",
            source_documents=[],
            confidence_score=0.0
        )
    
    def _create_error_result(self, disease_name: str, error_msg: str) -> TextRAGResult:
        """Create an error result"""
        return TextRAGResult(
            disease_name=disease_name,
            symptoms=f"Error retrieving information: {error_msg}",
            prevention="Unable to provide preventive measures due to system error.",
            treatment="Unable to provide treatment information due to system error.",
            additional_info="System error occurred during information retrieval.",
            source_documents=[],
            confidence_score=0.0
        )
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("TextRAG: MongoDB connection closed")