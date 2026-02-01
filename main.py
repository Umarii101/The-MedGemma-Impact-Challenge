"""
MedGemma Impact Challenge - Main Demo Script
Demonstrates all three core capabilities of the healthcare AI system.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines import MultimodalPipeline
from schemas.outputs import SystemHealthCheck
from utils.memory import check_cuda_setup, get_memory_manager
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def system_health_check() -> SystemHealthCheck:
    """Perform system health check"""
    print_header("SYSTEM HEALTH CHECK")
    
    cuda_info = check_cuda_setup()
    memory_mgr = get_memory_manager()
    
    health = SystemHealthCheck(
        gpu_available=cuda_info["cuda_available"],
        gpu_name=cuda_info.get("device_name"),
        gpu_memory_allocated=cuda_info.get("memory_stats", {}).get("allocated_gb"),
        gpu_memory_reserved=cuda_info.get("memory_stats", {}).get("reserved_gb"),
        models_loaded=[],
        system_ready=cuda_info["cuda_available"],
        warnings=[]
    )
    
    print(f"✓ CUDA Available: {health.gpu_available}")
    if health.gpu_available:
        print(f"✓ GPU: {health.gpu_name}")
        print(f"✓ CUDA Version: {cuda_info['cuda_version']}")
        print(f"✓ PyTorch Version: {cuda_info['pytorch_version']}")
        if health.gpu_memory_allocated:
            print(f"✓ GPU Memory Allocated: {health.gpu_memory_allocated:.2f} GB")
    else:
        print("⚠ WARNING: CUDA not available - will run on CPU (VERY SLOW)")
        health.warnings.append("CUDA not available - performance will be severely degraded")
    
    print(f"\n✓ System Ready: {health.system_ready}")
    
    return health


def demo_clinical_text_analysis(pipeline: MultimodalPipeline):
    """Demo 1: Clinical Text Understanding"""
    print_header("DEMO 1: CLINICAL TEXT ANALYSIS")
    
    # Example clinical note
    clinical_note = """
    Patient: 67-year-old male
    
    Chief Complaint: Persistent cough for 3 weeks with occasional shortness of breath
    
    History of Present Illness:
    Patient reports non-productive cough that started 3 weeks ago. Initially mild but has
    progressively worsened. Reports occasional dyspnea on exertion, especially when climbing
    stairs. Denies fever, chest pain, or hemoptysis. No recent travel or sick contacts.
    
    Past Medical History:
    - Hypertension (controlled on lisinopril)
    - Type 2 Diabetes (metformin)
    - Former smoker (quit 10 years ago, 30 pack-year history)
    
    Current Medications:
    - Lisinopril 20mg daily
    - Metformin 1000mg twice daily
    - Aspirin 81mg daily
    
    Physical Exam:
    Vitals: BP 142/88, HR 82, RR 18, SpO2 96% on room air, Temp 98.4°F
    General: Alert, no acute distress
    Lungs: Bilateral scattered wheezes, no crackles
    Heart: Regular rate and rhythm
    """
    
    print("Input Clinical Note:")
    print("-" * 80)
    print(clinical_note)
    print("-" * 80)
    
    logger.info("Running clinical text analysis...")
    
    try:
        result = pipeline.analyze_clinical_text(
            clinical_note=clinical_note,
            patient_age=67
        )
        
        print("\n📊 ANALYSIS RESULTS:\n")
        print(f"Summary:\n{result.summary}\n")
        
        print(f"Risk Level: {result.risk_level.value}")
        print(f"Confidence: {result.confidence:.0%}\n")
        
        if result.key_findings:
            print("Key Findings:")
            for finding in result.key_findings:
                print(f"  • {finding}")
        
        if result.extracted_symptoms:
            print(f"\nExtracted Symptoms: {', '.join(result.extracted_symptoms)}")
        
        if result.extracted_conditions:
            print(f"Extracted Conditions: {', '.join(result.extracted_conditions)}")
        
        if result.medications_mentioned:
            print(f"Medications: {', '.join(result.medications_mentioned)}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
        
        print(f"\n{result.safety_disclaimer}")
        
        # Save as JSON
        output_path = Path(__file__).parent / "demo_text_output.json"
        output_path.write_text(result.model_dump_json(indent=2), encoding='utf-8')
        print(f"\n✓ Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in clinical text analysis: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")


def demo_image_analysis(pipeline: MultimodalPipeline):
    """Demo 2: Medical Image Understanding (Assistive)"""
    print_header("DEMO 2: MEDICAL IMAGE ANALYSIS (ASSISTIVE)")
    
    print("⚠️  NOTE: This demo requires a medical image file.")
    print("   Since no image is provided, we'll demonstrate the API structure.\n")
    
    # Show how to use the image analysis
    code_example = """
# Example usage with actual image:
from PIL import Image

# Load medical image
image = Image.open("chest_xray.jpg")

# Analyze image
result = pipeline.analyze_image(
    image=image,
    image_type="Chest X-Ray",
    include_quality_check=True
)

print(f"Image Type: {result.image_type}")
print(f"Quality: {result.quality_assessment}")
print(f"Confidence: {result.confidence:.0%}")

print("\\nVisual Observations:")
for obs in result.visual_observations:
    print(f"  • {obs}")

print(f"\\n{result.safety_disclaimer}")
"""
    
    print("Code Example:")
    print("-" * 80)
    print(code_example)
    print("-" * 80)
    
    print("\n📋 Image Analysis Features:")
    print("  ✓ Feature extraction using RAD-DINO or CLIP")
    print("  ✓ Image quality assessment")
    print("  ✓ Non-diagnostic visual observations")
    print("  ✓ Confidence scoring")
    print("  ✓ Safety disclaimers enforced")
    
    print("\n⚠️  IMPORTANT: All image analysis is assistive only.")
    print("   Results require validation by qualified radiologist.")


def demo_multimodal_analysis(pipeline: MultimodalPipeline):
    """Demo 3: Integrated Text + Image Analysis"""
    print_header("DEMO 3: MULTIMODAL ANALYSIS (Text + Image)")
    
    print("⚠️  NOTE: This demo requires both clinical note and medical image.")
    print("   Since no image is provided, we'll demonstrate the API structure.\n")
    
    code_example = """
# Example usage with clinical note + image:
from PIL import Image

clinical_note = '''
Patient presents with persistent cough and dyspnea.
Past medical history: COPD, hypertension.
Chest X-ray ordered for evaluation.
'''

image = Image.open("chest_xray.jpg")

# Integrated multimodal analysis
result = pipeline.analyze_with_image(
    clinical_note=clinical_note,
    image=image,
    image_type="Chest X-Ray",
    patient_age=65,
    integrate_findings=True  # Use LLM to integrate findings
)

print(f"Clinical Summary:\\n{result.clinical_summary}\\n")

print("Integrated Findings:")
for finding in result.integrated_findings:
    print(f"  • {finding}")

print(f"\\nOverall Risk: {result.overall_risk_level.value}")
print(f"Confidence: {result.overall_confidence:.0%}")

print(f"\\nClinical Reasoning:\\n{result.clinical_reasoning}")

print("\\nRecommended Next Steps:")
for step in result.next_steps:
    print(f"  • {step}")

print(f"\\n{result.safety_disclaimer}")
"""
    
    print("Code Example:")
    print("-" * 80)
    print(code_example)
    print("-" * 80)
    
    print("\n📋 Multimodal Analysis Features:")
    print("  ✓ Combines clinical text + medical images")
    print("  ✓ MedGemma-powered reasoning across modalities")
    print("  ✓ Integrated risk assessment")
    print("  ✓ Correlated findings identification")
    print("  ✓ Unified clinical summary")
    print("  ✓ Structured JSON output")
    
    print("\n🎯 Use Cases:")
    print("  • Emergency triage support")
    print("  • Pre-visit preparation")
    print("  • Clinical handoff documentation")
    print("  • Educational case reviews")


def main():
    """Main demo function"""
    print("\n" + "=" * 80)
    print("  MedGemma Impact Challenge - Healthcare AI Backend Demo")
    print("  Production-Quality Local Inference System")
    print("=" * 80)
    
    # Step 1: System health check
    health = system_health_check()
    
    if not health.system_ready:
        print("\n⚠️  System not ready. Please check CUDA setup.")
        print("   The system will still run but performance will be degraded.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Step 2: Initialize pipeline
    print_header("INITIALIZING MULTIMODAL PIPELINE")
    print("Loading models... (this may take 1-2 minutes)")
    
    try:
        pipeline = MultimodalPipeline(
            use_8bit_llm=True,  # Use 8-bit for RTX 3080
            image_model_type="clip"  # CLIP is more widely available
        )
        
        memory_mgr = get_memory_manager()
        memory_mgr.log_memory_usage("After pipeline initialization")
        
        print("✓ Pipeline initialized successfully!")
        
        # Run demos
        demo_clinical_text_analysis(pipeline)
        demo_image_analysis(pipeline)
        demo_multimodal_analysis(pipeline)
        
        # Cleanup
        print_header("CLEANUP")
        pipeline.cleanup()
        memory_mgr.clear_cache()
        print("✓ Resources freed")
        
        # Final summary
        print_header("DEMO COMPLETE")
        print("✓ Clinical Text Analysis: Demonstrated")
        print("✓ Image Analysis API: Shown")
        print("✓ Multimodal Integration: Explained")
        
        print("\n📁 Output Files:")
        print("  • demo_text_output.json - Clinical text analysis results")
        
        print("\n🎯 Next Steps:")
        print("  1. Add medical images to test image analysis")
        print("  2. Try multimodal analysis with real cases")
        print("  3. Integrate into clinical workflow")
        print("  4. Collect feedback from healthcare professionals")
        
        print("\n⚠️  IMPORTANT REMINDERS:")
        print("  • This system is ASSISTIVE ONLY")
        print("  • All outputs require clinical validation")
        print("  • Not FDA approved or for diagnostic use")
        print("  • For research and demonstration purposes")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Check logs for details.")
    
    print("\n" + "=" * 80)
    print("  Thank you for using MedGemma Healthcare AI Backend!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
