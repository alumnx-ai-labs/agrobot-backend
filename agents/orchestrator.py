from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import logging
from typing import Dict, Any
from models.schemas import WorkflowState, WorkflowStatus
from tools.image_rag import ImageRAGTool
from tools.image_classification import ImageClassificationTool
from tools.text_rag import TextRAGTool
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

class CropDiseaseOrchestrator:
    """Main orchestrator for the crop disease analysis workflow"""
    
    def __init__(self):
        """Initialize the orchestrator with tools and workflow"""
        # Initialize tools
        self.image_rag_tool = ImageRAGTool()
        self.image_classification_tool = ImageClassificationTool()
        self.text_rag_tool = TextRAGTool()
        
        # Create and compile workflow
        self.workflow = self._create_workflow()
        
        logger.info("CropDiseaseOrchestrator initialized")
    
    async def execute_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Orchestrator: Starting workflow execution")
            logger.info(f"Orchestrator: Initial state - crop_type: {initial_state['crop_type']}, sme_advisor: {initial_state['sme_advisor']}")
            
            # No config needed without memory - each request is independent
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"Orchestrator: Workflow completed with status: {final_state['workflow_status']}")
            logger.info(f"Orchestrator: Final disease class: {final_state.get('final_disease_class')}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Orchestrator: Workflow execution failed: {str(e)}")
            logger.error(f"Orchestrator: Exception type: {type(e).__name__}")
            return {
                "image_data": initial_state["image_data"],
                "crop_type": initial_state["crop_type"],
                "sme_advisor": initial_state["sme_advisor"],
                "image_rag_results": [],
                "image_classification_result": {},
                "final_disease_class": None,
                "text_rag_results": None,
                "workflow_status": WorkflowStatus.ERROR.value,
                "error_message": f"Workflow execution failed: {str(e)}"
            }
    
    def _create_workflow(self) -> CompiledStateGraph:
        """Create and compile the LangGraph workflow"""
        logger.info("Orchestrator: Creating workflow graph")
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("image_rag", self._image_rag_node)
        workflow.add_node("image_classification", self._image_classification_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("text_rag", self._text_rag_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges
        workflow.add_edge("orchestrator", "image_rag")
        workflow.add_edge("orchestrator", "image_classification")
        workflow.add_edge("image_rag", "decision")
        workflow.add_edge("image_classification", "decision")
        
        # Add conditional edge
        workflow.add_conditional_edges(
            "decision",
            self._should_process_text_rag,
            {
                "text_rag": "text_rag",
                "end": END
            }
        )
        
        workflow.add_edge("text_rag", END)
        
        # Compile without memory - no checkpointer parameter
        logger.info("Orchestrator: Workflow graph created and compiled without memory")
        return workflow.compile()
    
    async def _orchestrator_node(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Orchestrator: Entering orchestrator node")
        logger.info(f"Orchestrator: Processing crop_type: {state['crop_type']}")
        logger.info(f"Orchestrator: SME advisor: {state['sme_advisor'] or 'None (optional)'}")
        
        logger.info("Orchestrator: Orchestrator node completed - starting parallel processing")
        return {"workflow_status": WorkflowStatus.PROCESSING.value}
    
    async def _image_rag_node(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Orchestrator: Entering ImageRAG node")
        
        try:
            logger.info("Orchestrator: Starting ImageRAG query")
            results = await self.image_rag_tool.query(state["image_data"], state["crop_type"], top_k=5)
            
            image_rag_results = [
                {
                    "disease_name": r.disease_name,
                    "confidence": r.confidence,
                    "image_url": r.image_url,
                    "description": r.description
                } for r in results
            ]
            
            logger.info(f"Orchestrator: ImageRAG raw results count: {len(results)}")
            logger.info(f"Orchestrator: ImageRAG processed results count: {len(image_rag_results)}")
            logger.info(f"Orchestrator: ImageRAG completed - found {len(results)} similar diseases")
            
            for i, result in enumerate(results[:5]):
                logger.info(f"Orchestrator: ImageRAG result {i+1}: {result.disease_name} (confidence: {result.confidence:.3f})")
            
            return {"image_rag_results": image_rag_results}
            
        except Exception as e:
            logger.error(f"Orchestrator: ImageRAG error: {str(e)}")
            logger.error(f"Orchestrator: ImageRAG exception type: {type(e).__name__}")
            return {
                "error_message": f"ImageRAG processing failed: {str(e)}",
                "workflow_status": WorkflowStatus.ERROR.value
            }
    
    async def _image_classification_node(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Orchestrator: Entering ImageClassification node")
        
        try:
            logger.info("Orchestrator: Starting ImageClassification prediction")
            result = await self.image_classification_tool.predict(state["image_data"], state["crop_type"])
            
            image_classification_result = {
                "disease_name": result.disease_name,
                "confidence": result.confidence,
                "description": result.description
            }
            logger.info(f"Orchestrator: ImageClassification completed - predicted {result.disease_name} with {result.confidence:.3f} confidence")
            
            return {"image_classification_result": image_classification_result}
            
        except Exception as e:
            logger.error(f"Orchestrator: ImageClassification error: {str(e)}")
            logger.error(f"Orchestrator: ImageClassification exception type: {type(e).__name__}")
            return {
                "error_message": f"Image classification failed: {str(e)}",
                "workflow_status": WorkflowStatus.ERROR.value
            }
    
    async def _decision_node(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Orchestrator: Entering decision node")
        
        try:
            logger.info("Orchestrator: Analyzing results in decision node")
            
            if state.get("error_message"):
                logger.warning(f"Orchestrator: Error detected in previous steps: {state['error_message']}")
                return {}
            
            rag_results = state.get("image_rag_results", [])
            classification_result = state.get("image_classification_result", {})
            
            logger.info(f"Orchestrator: Decision analysis - RAG results count: {len(rag_results)}")
            logger.info(f"Orchestrator: Decision analysis - Classification available: {bool(classification_result)}")
            
            if not rag_results or not classification_result:
                error_msg = "Missing required analysis results"
                logger.error(f"Orchestrator: {error_msg}")
                return {
                    "error_message": error_msg,
                    "workflow_status": WorkflowStatus.ERROR.value
                }
            
            predicted_disease = classification_result["disease_name"]
            logger.info(f"Orchestrator: Classification predicted: {predicted_disease}")
            
            rag_diseases = [r["disease_name"] for r in rag_results]
            classification_matches = rag_diseases.count(predicted_disease)
            
            logger.info(f"Orchestrator: Classification '{predicted_disease}' matches {classification_matches} RAG results")
            logger.info(f"Orchestrator: RAG diseases found: {set(rag_diseases)}")
            
            if classification_matches >= 3:
                logger.info("Orchestrator: Classification match criteria met (≥3 matches)")
                return {
                    "final_disease_class": predicted_disease,
                    "workflow_status": WorkflowStatus.CLASSIFICATION_MATCH.value
                }
            else:
                disease_counts = {}
                for disease in rag_diseases:
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
                
                max_count = max(disease_counts.values()) if disease_counts else 0
                consensus_disease = max(disease_counts.items(), key=lambda x: x[1])[0] if disease_counts else None
                
                logger.info(f"Orchestrator: RAG consensus analysis - disease counts: {disease_counts}")
                logger.info(f"Orchestrator: RAG consensus - max count {max_count} for '{consensus_disease}'")
                
                if max_count >= 4:
                    logger.info("Orchestrator: RAG consensus criteria met (≥4 matches)")
                    return {
                        "final_disease_class": consensus_disease,
                        "workflow_status": WorkflowStatus.RAG_CONSENSUS.value
                    }
                else:
                    logger.info("Orchestrator: No clear consensus - returning top 5 possibilities")
                    return {"workflow_status": WorkflowStatus.UNCERTAIN.value}
                    
        except Exception as e:
            logger.error(f"Orchestrator: Decision error: {str(e)}")
            logger.error(f"Orchestrator: Decision exception type: {type(e).__name__}")
            return {
                "error_message": f"Decision processing failed: {str(e)}",
                "workflow_status": WorkflowStatus.ERROR.value
            }
    
    async def _text_rag_node(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Orchestrator: Entering TextRAG node")
        
        try:
            if state.get("final_disease_class"):
                logger.info(f"Orchestrator: Querying TextRAG for disease: {state['final_disease_class']}")
                logger.info(f"Orchestrator: Using SME advisor: {state['sme_advisor'] or 'None (optional)'}")
                
                result = await self.text_rag_tool.query(
                    disease_name=state["final_disease_class"], 
                    sme_advisor=state["sme_advisor"],
                    crop_type=state["crop_type"]
                )
                
                logger.info(f"Orchestrator: TextRAG completed - retrieved info with confidence {result.confidence_score:.3f}")
                logger.info(f"Orchestrator: TextRAG found {len(result.source_documents)} source documents")
                
                return {
                    "text_rag_results": {
                        "disease_name": result.disease_name,
                        "symptoms": result.symptoms,
                        "prevention": result.prevention,
                        "treatment": result.treatment,
                        "additional_info": result.additional_info,
                        "source_documents": result.source_documents,
                        "confidence_score": result.confidence_score
                    },
                    "workflow_status": WorkflowStatus.COMPLETED.value
                }
            else:
                logger.warning("Orchestrator: No final disease class determined, skipping TextRAG")
                return {}
                
        except Exception as e:
            logger.error(f"Orchestrator: TextRAG error: {str(e)}")
            logger.error(f"Orchestrator: TextRAG exception type: {type(e).__name__}")
            return {
                "error_message": f"TextRAG processing failed: {str(e)}",
                "workflow_status": WorkflowStatus.ERROR.value
            }
    
    def _should_process_text_rag(self, state: WorkflowState) -> str:
        status = state.get("workflow_status")
        logger.info(f"Orchestrator: Conditional check - current status: {status}")
        
        if status in [WorkflowStatus.CLASSIFICATION_MATCH.value, WorkflowStatus.RAG_CONSENSUS.value]:
            logger.info("Orchestrator: Proceeding to TextRAG")
            return "text_rag"
        
        logger.info("Orchestrator: Ending workflow without TextRAG")
        return "end"