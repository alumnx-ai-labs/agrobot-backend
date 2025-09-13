from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
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
    
    async def execute_workflow(self, initial_state: WorkflowState) -> Dict[str, Any]:
        """Execute the complete workflow"""
        try:
            logger.info("Orchestrator: Starting workflow execution")
            
            config = {"configurable": {"thread_id": "crop_analysis_session"}}
            final_state = await self.workflow.ainvoke(initial_state, config)
            
            logger.info(f"Orchestrator: Workflow completed with status: {final_state.get('workflow_status')}")
            return final_state
            
        except Exception as e:
            logger.error(f"Orchestrator: Workflow execution failed: {str(e)}")
            return {
                **initial_state,
                "error_message": f"Workflow execution failed: {str(e)}",
                "workflow_status": WorkflowStatus.ERROR.value
            }
    
    def _create_workflow(self) -> CompiledStateGraph:
        """Create and compile the LangGraph workflow"""
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
        
        # Add memory
        memory = MemorySaver()
        
        return workflow.compile(checkpointer=memory)
    
    async def _orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """Main orchestrator node - initiates parallel processing"""
        logger.info("Orchestrator: Starting workflow")
        state["workflow_status"] = WorkflowStatus.PROCESSING.value
        return state
    
    async def _image_rag_node(self, state: WorkflowState) -> WorkflowState:
        """ImageRAG processing node"""
        try:
            logger.info("Orchestrator: Processing ImageRAG")
            results = await self.image_rag_tool.query(state["image_data"], state["crop_type"])
            
            state["image_rag_results"] = [
                {
                    "disease_name": r.disease_name,
                    "confidence": r.confidence,
                    "image_url": r.image_url,
                    "description": r.description
                } for r in results
            ]
            logger.info(f"Orchestrator: ImageRAG found {len(results)} similar diseases")
            
        except Exception as e:
            logger.error(f"Orchestrator: ImageRAG error: {str(e)}")
            state["error_message"] = f"ImageRAG processing failed: {str(e)}"
            state["workflow_status"] = WorkflowStatus.ERROR.value
        
        return state
    
    async def _image_classification_node(self, state: WorkflowState) -> WorkflowState:
        """Image Classification processing node"""
        try:
            logger.info("Orchestrator: Processing ImageClassification")
            result = await self.image_classification_tool.predict(state["image_data"], state["crop_type"])
            
            state["image_classification_result"] = {
                "disease_name": result.disease_name,
                "confidence": result.confidence,
                "description": result.description
            }
            logger.info(f"Orchestrator: ImageClassification predicted {result.disease_name} with {result.confidence:.2f} confidence")
            
        except Exception as e:
            logger.error(f"Orchestrator: ImageClassification error: {str(e)}")
            state["error_message"] = f"Image classification failed: {str(e)}"
            state["workflow_status"] = WorkflowStatus.ERROR.value
        
        return state
    
    async def _decision_node(self, state: WorkflowState) -> WorkflowState:
        """Decision node - implements the comparison logic"""
        try:
            logger.info("Orchestrator: Analyzing results in decision node")
            
            if state.get("error_message"):
                return state
            
            rag_results = state.get("image_rag_results", [])
            classification_result = state.get("image_classification_result", {})
            
            if not rag_results or not classification_result:
                state["error_message"] = "Missing required analysis results"
                state["workflow_status"] = WorkflowStatus.ERROR.value
                return state
            
            predicted_disease = classification_result["disease_name"]
            
            # Check if ImageClassification matches ≥3 results in ImageRAG
            rag_diseases = [r["disease_name"] for r in rag_results]
            classification_matches = rag_diseases.count(predicted_disease)
            
            logger.info(f"Orchestrator: Classification '{predicted_disease}' matches {classification_matches} RAG results")
            
            if classification_matches >= 3:
                state["final_disease_class"] = predicted_disease
                state["workflow_status"] = WorkflowStatus.CLASSIFICATION_MATCH.value
                logger.info("Orchestrator: Classification match criteria met")
            else:
                # Check if ≥4 results in ImageRAG agree on a single disease
                disease_counts = {}
                for disease in rag_diseases:
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
                
                max_count = max(disease_counts.values()) if disease_counts else 0
                consensus_disease = max(disease_counts.items(), key=lambda x: x[1])[0] if disease_counts else None
                
                logger.info(f"Orchestrator: RAG consensus - max count {max_count} for '{consensus_disease}'")
                
                if max_count >= 4:
                    state["final_disease_class"] = consensus_disease
                    state["workflow_status"] = WorkflowStatus.RAG_CONSENSUS.value
                    logger.info("Orchestrator: RAG consensus criteria met")
                else:
                    state["workflow_status"] = WorkflowStatus.UNCERTAIN.value
                    logger.info("Orchestrator: No clear consensus - returning top 5 possibilities")
        
        except Exception as e:
            logger.error(f"Orchestrator: Decision error: {str(e)}")
            state["error_message"] = f"Decision processing failed: {str(e)}"
            state["workflow_status"] = WorkflowStatus.ERROR.value
        
        return state
    
    async def _text_rag_node(self, state: WorkflowState) -> WorkflowState:
        """TextRAG processing node"""
        try:
            if state["final_disease_class"]:
                logger.info(f"Orchestrator: Querying TextRAG for disease: {state['final_disease_class']}")
                
                result = await self.text_rag_tool.query(
                    disease_name=state["final_disease_class"], 
                    sme_advisor=state["sme_advisor"],
                    crop_type=state["crop_type"]
                )
                
                state["text_rag_results"] = {
                    "disease_name": result.disease_name,
                    "symptoms": result.symptoms,
                    "prevention": result.prevention,
                    "treatment": result.treatment,
                    "additional_info": result.additional_info,
                    "source_documents": result.source_documents,
                    "confidence_score": result.confidence_score
                }
                
                state["workflow_status"] = WorkflowStatus.COMPLETED.value
                logger.info(f"Orchestrator: TextRAG retrieved preventive measures with confidence {result.confidence_score:.2f}")
        
        except Exception as e:
            logger.error(f"Orchestrator: TextRAG error: {str(e)}")
            state["error_message"] = f"TextRAG processing failed: {str(e)}"
            state["workflow_status"] = WorkflowStatus.ERROR.value
        
        return state
    
    def _should_process_text_rag(self, state: WorkflowState) -> str:
        """Conditional edge function"""
        status = state.get("workflow_status")
        if status in [WorkflowStatus.CLASSIFICATION_MATCH.value, WorkflowStatus.RAG_CONSENSUS.value]:
            return "text_rag"
        return "end"