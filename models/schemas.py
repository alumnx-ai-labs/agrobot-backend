from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    PROCESSING = "processing"
    CLASSIFICATION_MATCH = "classification_match"
    RAG_CONSENSUS = "rag_consensus" 
    UNCERTAIN = "uncertain"
    COMPLETED = "completed"
    ERROR = "error"

class WorkflowState(BaseModel):
    """State model for the crop disease analysis workflow"""
    # Input data
    image_data: str  # Base64 encoded image
    crop_type: str
    sme_advisor: str
    
    # Analysis results
    image_rag_results: List[Dict[str, Any]] = []
    image_classification_result: Dict[str, Any] = {}
    final_disease_class: Optional[str] = None
    text_rag_results: Optional[Dict[str, Any]] = None
    
    # Workflow control
    workflow_status: str = WorkflowStatus.PROCESSING.value
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True