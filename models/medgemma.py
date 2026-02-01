"""
MedGemma model loader and inference engine.
Primary LLM for clinical reasoning and text understanding.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, List
import logging
import json
import re

from utils.memory import get_memory_manager
from utils.safety import get_safety_framer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedGemmaEngine:
    """
    MedGemma 4B inference engine for clinical text understanding.
    Optimized for RTX 3080 (10GB VRAM).
    """
    
    MODEL_NAME = "google/medgemma-4b-it"  # Primary model (4B fits better on 10GB GPU, -it = instruction-tuned)
    # Fallback models: Gemma-3-4B or Llama-2-7B
    
    def __init__(self, model_name: Optional[str] = None, use_8bit: bool = True):
        """
        Initialize MedGemma engine.
        
        Args:
            model_name: Hugging Face model identifier (defaults to MedGemma 7B)
            use_8bit: Use 8-bit quantization for memory efficiency
        """
        self.model_name = model_name or self.MODEL_NAME
        self.use_8bit = use_8bit
        self.memory_manager = get_memory_manager()
        self.safety_framer = get_safety_framer()
        
        self.model = None
        self.tokenizer = None
        self.device = self.memory_manager.device
        
        logger.info(f"Initializing MedGemma Engine with model: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load MedGemma model and tokenizer with memory optimization"""
        try:
            self.memory_manager.log_memory_usage("Before MedGemma load")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model loading kwargs for RTX 3080
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Use 8-bit quantization for 10GB GPU
            if self.use_8bit:
                logger.info("Loading model with 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                model_kwargs.update({
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                })
            
            # Load model
            logger.info(f"Loading model {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Optimize for inference
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.memory_manager.log_memory_usage("After MedGemma load")
            logger.info("MedGemma model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading MedGemma: {e}")
            logger.warning("Attempting fallback to Gemma-2-7B-IT...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model if MedGemma unavailable"""
        fallback_models = [
            "google/gemma-3-4b-it",
            "meta-llama/Llama-2-7b-hf",
        ]
        
        for fallback in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback}")
                self.model_name = fallback
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Prepare model loading with proper quantization config
                model_kwargs = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "device_map": "auto",
                }
                
                if self.use_8bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback,
                    **model_kwargs
                )
                
                self.model.eval()
                logger.info(f"Successfully loaded fallback model: {fallback}")
                return
                
            except Exception as e:
                logger.error(f"Failed to load {fallback}: {e}")
                continue
        
        raise RuntimeError("Failed to load MedGemma or any fallback model")
    
    def generate_clinical_summary(
        self,
        clinical_note: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> Dict:
        """
        Generate structured clinical summary from note.
        
        Args:
            clinical_note: Raw clinical note text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary with summary, findings, and extracted entities
        """
        prompt = self._create_summary_prompt(clinical_note)
        
        # Generate
        output_text = self._generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Parse output
        parsed = self._parse_clinical_output(output_text)
        
        # Apply safety framing
        if "summary" in parsed:
            parsed["summary"] = self.safety_framer.frame_output(parsed["summary"])
        
        return parsed
    
    def extract_clinical_entities(self, clinical_note: str) -> Dict[str, List[str]]:
        """
        Extract symptoms, conditions, and medications from clinical note.
        
        Args:
            clinical_note: Clinical note text
            
        Returns:
            Dictionary with extracted entities
        """
        prompt = self._create_extraction_prompt(clinical_note)
        
        output_text = self._generate(
            prompt,
            max_new_tokens=384,
            temperature=0.1  # Low temperature for extraction
        )
        
        # Parse JSON output
        entities = self._parse_entity_output(output_text)
        return entities
    
    def generate_recommendations(
        self,
        clinical_summary: str,
        risk_level: str = "Unknown"
    ) -> List[str]:
        """
        Generate clinical recommendations based on summary.
        
        Args:
            clinical_summary: Clinical summary text
            risk_level: Estimated risk level
            
        Returns:
            List of recommendations
        """
        prompt = self._create_recommendation_prompt(clinical_summary, risk_level)
        
        output_text = self._generate(
            prompt,
            max_new_tokens=256,
            temperature=0.4
        )
        
        # Parse recommendations
        recommendations = self._parse_recommendations(output_text)
        
        # Validate recommendations
        recommendations = self.safety_framer.validate_recommendations(recommendations)
        
        return recommendations
    
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Core generation method"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _create_summary_prompt(self, clinical_note: str) -> str:
        """Create prompt for clinical summarization"""
        prompt = f"""You are a medical AI assistant helping clinicians. Analyze this clinical note and provide a structured summary.

Clinical Note:
{clinical_note}

Provide a JSON response with:
1. "summary": Brief clinical summary (2-3 sentences)
2. "key_findings": List of important observations
3. "risk_indicators": Any concerning findings

Remember: You are providing assistive analysis, not making diagnoses. Use appropriate medical terminology but frame findings as observations.

Response (JSON format):"""
        return prompt
    
    def _create_extraction_prompt(self, clinical_note: str) -> str:
        """Create prompt for entity extraction"""
        prompt = f"""Extract clinical entities from this note. Provide a JSON response.

Clinical Note:
{clinical_note}

Extract and return JSON with these fields:
- "symptoms": List of symptoms mentioned
- "conditions": List of conditions/diagnoses mentioned (historical)
- "medications": List of medications referenced
- "procedures": List of procedures mentioned

Response (JSON format):"""
        return prompt
    
    def _create_recommendation_prompt(self, summary: str, risk_level: str) -> str:
        """Create prompt for generating recommendations"""
        prompt = f"""Based on this clinical summary, suggest next steps for the clinical team to consider.

Summary: {summary}
Risk Level: {risk_level}

Provide 3-5 recommendations as a JSON array. Frame as suggestions for consideration, not directives.
Use phrases like "Consider...", "May want to evaluate...", "Suggest reviewing..."

Response (JSON array of recommendations):"""
        return prompt
    
    def _parse_clinical_output(self, text: str) -> Dict:
        """Parse JSON output from clinical summary"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse as text
        return {
            "summary": text[:500],
            "key_findings": [],
            "risk_indicators": []
        }
    
    def _parse_entity_output(self, text: str) -> Dict[str, List[str]]:
        """Parse entity extraction output"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback
        return {
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "procedures": []
        }
    
    def _parse_recommendations(self, text: str) -> List[str]:
        """Parse recommendations from output"""
        try:
            # Try JSON array
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                recs = json.loads(json_match.group())
                if isinstance(recs, list):
                    return recs
        except json.JSONDecodeError:
            pass
        
        # Fallback: split by newlines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Clean up markdown/numbering
        recs = []
        for line in lines:
            cleaned = re.sub(r'^[\d\-\*\.\)]+\s*', '', line)
            if len(cleaned) > 10:
                recs.append(cleaned)
        
        return recs[:5]  # Limit to 5 recommendations
    
    def cleanup(self):
        """Clean up model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.memory_manager.clear_cache()
        logger.info("MedGemma engine cleaned up")
