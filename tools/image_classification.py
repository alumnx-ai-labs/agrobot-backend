import os
import json
import logging
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Result structure for image classification"""
    disease_name: str
    confidence: float
    description: str
    crop_type: str
    all_predictions: Dict[str, float]

class ImageClassificationTool:
    """
    Image Classification tool using MobileNet models for crop disease detection
    Supports multiple models based on crop type
    """
    
    def __init__(self):
        """Initialize the ImageClassificationTool"""
        self.models_base_path = "ml-models"
        self.models = {}  # Cache for loaded models
        self.class_labels = {}  # Cache for class labels
        
        # Default model configurations
        self.model_config = {
            "default": {
                "model_path": os.getenv("MOBILENET_MODEL_PATH", "ml-models/crop_disease_mobilenet.h5"),
                "labels_path": os.getenv("CLASS_LABELS_PATH", "ml-models/class_labels.json")
            }
        }
        
        # Image preprocessing parameters
        self.input_size = (224, 224)  # Standard MobileNet input size
        self.confidence_threshold = 0.5
        
        logger.info("ImageClassificationTool initialized")
    
    def _get_model_paths(self, crop_type: str) -> Dict[str, str]:
        """
        Get model and labels paths for specific crop type
        
        Args:
            crop_type: The crop type to get models for
            
        Returns:
            Dictionary with model_path and labels_path
        """
        # Normalize crop type for file naming
        crop_normalized = crop_type.lower().replace(" ", "_").replace("-", "_")
        
        # Check if crop-specific model exists
        crop_model_path = f"{self.models_base_path}/{crop_normalized}_mobilenet.h5"
        crop_labels_path = f"{self.models_base_path}/{crop_normalized}_labels.json"
        
        if os.path.exists(crop_model_path) and os.path.exists(crop_labels_path):
            logger.info(f"ImageClassification: Using crop-specific model for {crop_type}")
            return {
                "model_path": crop_model_path,
                "labels_path": crop_labels_path
            }
        
        # Fall back to default model
        logger.info(f"ImageClassification: Using default model for {crop_type}")
        return self.model_config["default"]
    
    def _load_model(self, model_path: str) -> Optional[keras.Model]:
        """
        Load TensorFlow/Keras model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"ImageClassification: Model file not found: {model_path}")
                return None
            
            model = keras.models.load_model(model_path)
            logger.info(f"ImageClassification: Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"ImageClassification: Failed to load model from {model_path}: {str(e)}")
            return None
    
    def _load_class_labels(self, labels_path: str) -> Optional[List[str]]:
        """
        Load class labels from JSON file
        
        Args:
            labels_path: Path to the labels JSON file
            
        Returns:
            List of class labels or None if loading fails
        """
        try:
            if not os.path.exists(labels_path):
                logger.warning(f"ImageClassification: Labels file not found: {labels_path}")
                return None
            
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            
            # Support different JSON structures
            if isinstance(labels_data, list):
                labels = labels_data
            elif isinstance(labels_data, dict):
                # Try common keys
                labels = labels_data.get('labels') or labels_data.get('classes') or list(labels_data.values())
            else:
                raise ValueError("Invalid labels file format")
            
            logger.info(f"ImageClassification: Loaded {len(labels)} class labels from {labels_path}")
            return labels
            
        except Exception as e:
            logger.error(f"ImageClassification: Failed to load labels from {labels_path}: {str(e)}")
            return None
    
    def _get_model_and_labels(self, crop_type: str) -> tuple[Optional[keras.Model], Optional[List[str]]]:
        """
        Get or load model and labels for specific crop type
        
        Args:
            crop_type: The crop type
            
        Returns:
            Tuple of (model, labels) or (None, None) if loading fails
        """
        # Check cache first
        cache_key = crop_type.lower()
        if cache_key in self.models and cache_key in self.class_labels:
            return self.models[cache_key], self.class_labels[cache_key]
        
        # Get paths for this crop type
        paths = self._get_model_paths(crop_type)
        
        # Load model and labels
        model = self._load_model(paths["model_path"])
        labels = self._load_class_labels(paths["labels_path"])
        
        if model is not None and labels is not None:
            # Cache the loaded model and labels
            self.models[cache_key] = model
            self.class_labels[cache_key] = labels
            logger.info(f"ImageClassification: Cached model and labels for {crop_type}")
        
        return model, labels
    
    def _preprocess_image(self, image_data: str) -> Optional[np.ndarray]:
        """
        Preprocess base64 image data for model input
        
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
            
            # Resize to model input size
            image = image.resize(self.input_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image)
            image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"ImageClassification: Image preprocessing failed: {str(e)}")
            return None
    
    async def predict(self, image_data: str, crop_type: str) -> ClassificationResult:
        """
        Predict disease class from image data
        
        Args:
            image_data: Base64 encoded image data
            crop_type: Type of crop for model selection
            
        Returns:
            ClassificationResult with prediction details
        """
        try:
            logger.info(f"ImageClassification: Starting prediction for crop type: {crop_type}")
            
            # Get model and labels
            model, labels = self._get_model_and_labels(crop_type)
            
            if model is None or labels is None:
                return self._create_error_result(
                    crop_type, 
                    f"Model or labels not available for crop type: {crop_type}"
                )
            
            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            if processed_image is None:
                return self._create_error_result(crop_type, "Failed to preprocess image")
            
            # Make prediction
            predictions = model.predict(processed_image, verbose=0)
            predictions = predictions[0]  # Remove batch dimension
            
            # Get top prediction
            max_idx = np.argmax(predictions)
            max_confidence = float(predictions[max_idx])
            predicted_disease = labels[max_idx]
            
            # Create all predictions dictionary
            all_predictions = {
                labels[i]: float(predictions[i]) 
                for i in range(len(labels))
            }
            
            # Sort by confidence
            all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"ImageClassification: Predicted '{predicted_disease}' with confidence {max_confidence:.3f}")
            
            # Check confidence threshold
            if max_confidence < self.confidence_threshold:
                logger.warning(f"ImageClassification: Low confidence prediction ({max_confidence:.3f})")
            
            return ClassificationResult(
                disease_name=predicted_disease,
                confidence=max_confidence,
                description=f"Predicted disease class for {crop_type} with {max_confidence:.1%} confidence",
                crop_type=crop_type,
                all_predictions=all_predictions
            )
            
        except Exception as e:
            logger.error(f"ImageClassification: Prediction failed: {str(e)}")
            return self._create_error_result(crop_type, f"Prediction failed: {str(e)}")
    
    def _create_error_result(self, crop_type: str, error_msg: str) -> ClassificationResult:
        """Create an error result"""
        return ClassificationResult(
            disease_name="classification_error",
            confidence=0.0,
            description=f"Classification error: {error_msg}",
            crop_type=crop_type,
            all_predictions={}
        )
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Get information about available models
        
        Returns:
            Dictionary mapping crop types to model availability
        """
        available_models = {}
        
        # Check default model
        default_paths = self.model_config["default"]
        available_models["default"] = (
            os.path.exists(default_paths["model_path"]) and 
            os.path.exists(default_paths["labels_path"])
        )
        
        # Check for crop-specific models in ml-models directory
        if os.path.exists(self.models_base_path):
            for file in os.listdir(self.models_base_path):
                if file.endswith("_mobilenet.h5"):
                    crop_name = file.replace("_mobilenet.h5", "")
                    labels_file = f"{crop_name}_labels.json"
                    labels_path = os.path.join(self.models_base_path, labels_file)
                    
                    available_models[crop_name] = os.path.exists(labels_path)
        
        return available_models
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self.models.clear()
        self.class_labels.clear()
        logger.info("ImageClassification: Model cache cleared")