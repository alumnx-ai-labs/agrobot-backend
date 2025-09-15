from enum import Enum
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from typing import Annotated
from operator import add

class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    PROCESSING = "processing"
    CLASSIFICATION_MATCH = "classification_match"
    RAG_CONSENSUS = "rag_consensus" 
    UNCERTAIN = "uncertain"
    COMPLETED = "completed"
    ERROR = "error"

class WorkflowState(TypedDict):
    """State model for the crop disease analysis workflow"""
    # Input data
    image_data: str  # Base64 encoded image
    crop_type: str
    sme_advisor: Optional[str]
    
    # Analysis results
    image_rag_results: Annotated[List[Dict[str, Any]], add]
    image_classification_result: Dict[str, Any]
    final_disease_class: Optional[str]
    text_rag_results: Optional[Dict[str, Any]]
    
    # Workflow control
    workflow_status: str
    error_message: Optional[str]