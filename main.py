from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
import uvicorn
from typing import Annotated, Optional
from PIL import Image
import io
import base64
import logging
from agents.orchestrator import CropDiseaseOrchestrator
from models.schemas import WorkflowState, WorkflowStatus
from dotenv import load_dotenv
from pydantic import BaseModel
from tools.text_rag import TextRAGTool

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Disease Analysis API",
    description="AI-powered crop disease detection and prevention recommendations",
    version="1.0.0"
)

# Initialize orchestrator
orchestrator = CropDiseaseOrchestrator()

# Initialize TextRAGTool
text_rag_tool = TextRAGTool()

def process_image(image_file: UploadFile) -> str:
    """Process uploaded image and convert to base64"""
    try:
        # Read image data
        image_data = image_file.file.read()
        
        # Validate image
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify it's a valid image
        except Exception:
            raise ValueError("Invalid image file")
        
        # Convert to base64
        return base64.b64encode(image_data).decode('utf-8')
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.post("/give-image")
async def analyze_crop_disease(
    image: Annotated[UploadFile, File(description="Image of the crop to analyze")],
    cropType: Annotated[str, Form(description="Type of crop (e.g., tomato, wheat, rice)")],
    smeAdvisor: Annotated[Optional[str], Form(description="Subject Matter Expert advisor name (optional)")] = None
):
    """
    Analyze crop disease from uploaded image using agentic AI workflow
    
    **Workflow:**
    1. Processes image through ImageRAG and ImageClassification in parallel
    2. Compares results to determine disease class confidence
    3. If confident, queries TextRAG for preventive measures
    4. Returns either specific disease info or top 5 possibilities
    """
    
    try:
        logger.info(f"New request: cropType={cropType}, smeAdvisor={smeAdvisor or 'None (optional)'}")
        
        # Validate inputs
        if not cropType.strip():
            raise HTTPException(status_code=400, detail="cropType is required")
        
        if not image.filename:
            raise HTTPException(status_code=400, detail="Image file is required")
        
        # Process image
        image_data = process_image(image)
        
        # Initialize workflow state
        initial_state = WorkflowState(
            image_data=image_data,
            crop_type=cropType.strip(),
            sme_advisor=smeAdvisor.strip() if smeAdvisor else None,
            image_rag_results=[],
            image_classification_result={},
            final_disease_class=None,
            text_rag_results=None,
            workflow_status=WorkflowStatus.PROCESSING.value,
            error_message=None
        )
        
        # Execute workflow through orchestrator
        final_state = await orchestrator.execute_workflow(initial_state)
        
        # Handle errors
        if final_state.get("error_message"):
            raise HTTPException(
                status_code=500, 
                detail=f"Workflow error: {final_state['error_message']}"
            )
        
        # Prepare response based on workflow status
        status = final_state.get("workflow_status")
        
        if status in [WorkflowStatus.CLASSIFICATION_MATCH.value, WorkflowStatus.RAG_CONSENSUS.value]:
            # Confident prediction with preventive measures
            response_data = {
                "status": "confident_prediction",
                "disease_class": final_state["final_disease_class"],
                "confidence_reason": "classification_match" if status == WorkflowStatus.CLASSIFICATION_MATCH.value else "rag_consensus",
                "disease_info": final_state.get("text_rag_results", {}),
                "classification_result": final_state["image_classification_result"],
                "similar_diseases": final_state["image_rag_results"][:5]
            }
            
        elif status == WorkflowStatus.UNCERTAIN.value:
            # Uncertain - return top 5 possibilities
            response_data = {
                "status": "uncertain_prediction",
                "message": "Unable to determine disease class with confidence",
                "top_possibilities": final_state["image_rag_results"][:5],
                "classification_result": final_state["image_classification_result"],
                "recommendation": "Consider consulting with multiple experts or obtaining additional images"
            }
            
        else:
            # Unexpected status
            response_data = {
                "status": "processing_incomplete",
                "workflow_status": status,
                "message": "Analysis could not be completed",
                "partial_results": {
                    "classification": final_state.get("image_classification_result"),
                    "similar_diseases": final_state.get("image_rag_results", [])[:5]
                }
            }
        
        logger.info(f"Request completed with status: {status}")
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# New Pydantic model for the request body
class DiseaseConfirmation(BaseModel):
    disease_class: str
    sme_advisor: Optional[str] = None
    crop_type: Optional[str] = None

@app.post("/confirm-disease")
async def confirm_disease(
    confirmation_data: Annotated[DiseaseConfirmation, Body(description="Disease and SME details to confirm")]
):
    """
    Confirms a disease and retrieves detailed information from the TextRAG system.
    """
    logger.info(f"New confirmation request for disease: {confirmation_data.disease_class}")
    try:
        # Call the TextRAG tool to get relevant information
        result = await text_rag_tool.query(
            disease_name=confirmation_data.disease_class,
            sme_advisor=confirmation_data.sme_advisor,
            crop_type=confirmation_data.crop_type
        )
        
        # Check if the query was successful
        if result.symptoms.startswith("Error"):
            raise HTTPException(
                status_code=500,
                detail=result.symptoms
            )
            
        return JSONResponse(content={
            "status": "success",
            "disease_info": {
                "disease_name": result.disease_name,
                "symptoms": result.symptoms,
                "prevention": result.prevention,
                "treatment": result.treatment,
                "additional_info": result.additional_info,
                "source_documents": result.source_documents,
                "confidence_score": result.confidence_score
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing confirmation request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Crop Disease Analysis API"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Crop Disease Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/give-image (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )