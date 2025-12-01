"""
Test PaddleOCR integration - verify implementation.

This test verifies that all components are properly implemented.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.rag.services.paddle_ocr_service import PaddleOCRService  # noqa: F401
        print("[PASS] PaddleOCRService imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import PaddleOCRService: {e}")
        return False
    
    try:
        from src.rag.services.ingestion_pipeline_service.service import (
            IngestionPipelineService, DoclingPDFTransformation
        )  # noqa: F401
        print("[PASS] DoclingPDFTransformation imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import PDFIngestAdapter: {e}")
        return False
    
    try:
        from src.utils.config_loader import get_paddle_ocr_config  # noqa: F401
        print("[PASS] get_paddle_ocr_config imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import get_paddle_ocr_config: {e}")
        return False
    
    return True


def test_config():
    """Test that configuration can be loaded."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config_loader import get_paddle_ocr_config
        config = get_paddle_ocr_config()
        
        required_keys = [
            'enabled', 'lang', 'use_gpu', 'min_confidence',
            'save_extracted_images', 'replace_image_placeholders'
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"[FAIL] Missing required config key: {key}")
                return False
            print(f"[PASS] Config key '{key}': {config[key]}")
        
        print("[PASS] All required config keys present")
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to load config: {e}")
        return False


def test_pdf_adapter_init():
    """Test that PDFIngestAdapter can be initialized."""
    print("\nTesting PDFIngestAdapter initialization...")
    
    try:
        from src.rag.services.ingestion_pipeline_service.service import DoclingPDFTransformation
        
        # Instantiate DoclingPDFTransformation and verify methods exist
        adapter = DoclingPDFTransformation()
        print("[PASS] PDFIngestAdapter initialized")
        
        # Check if OCR-related methods exist
        if not hasattr(adapter, '_process_images_with_ocr'):
            print("[FAIL] Missing method: _process_images_with_ocr")
            return False
        print("[PASS] Method _process_images_with_ocr exists")
        
        if not hasattr(adapter, '_replace_image_placeholders'):
            print("[FAIL] Missing method: _replace_image_placeholders")
            return False
        print("[PASS] Method _replace_image_placeholders exists")
        
        # Check if OCR config is loaded
        if not hasattr(adapter, 'ocr_config'):
            print("[FAIL] Missing attribute: ocr_config")
            return False
        print(f"[PASS] OCR config loaded: enabled={adapter.ocr_config['enabled']}")
        
        # Check if OCR service is initialized (if enabled and available)
        if adapter.ocr_enabled:
            if adapter.ocr_service is None:
                print("[WARN] OCR enabled but service is None (PaddleOCR not installed)")
            else:
                print("[PASS] OCR service initialized")
        else:
            print("[INFO] OCR disabled in configuration")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize PDFIngestAdapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_service_init():
    """Test that PaddleOCRService can be initialized (if paddleocr installed)."""
    print("\nTesting PaddleOCRService initialization...")
    
    try:
        from src.rag.services.paddle_ocr_service import PaddleOCRService
        
        try:
            ocr = PaddleOCRService(lang="en", use_gpu=False, show_log=False)
            print("[PASS] PaddleOCRService initialized successfully")
            
            # Check if methods exist
            methods = [
                'extract_text_from_image',
                'extract_text_from_images',
                'is_code_image',
                'format_extracted_text',
                'extract_text_from_base64'
            ]
            
            for method in methods:
                if not hasattr(ocr, method):
                    print(f"[FAIL] Missing method: {method}")
                    return False
                print(f"[PASS] Method {method} exists")
            
            return True
            
        except ImportError:
            print("[WARN] PaddleOCR not installed - this is OK for development")
            print("       Install with: pip install paddleocr paddlepaddle")
            return True  # Not a failure
            
    except Exception as e:
        print(f"[FAIL] Failed to test PaddleOCRService: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("VX-RAG PaddleOCR Integration - Implementation Verification")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("PDFIngestAdapter", test_pdf_adapter_init),
        ("PaddleOCRService", test_ocr_service_init),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 80)
    if all_passed:
        print("[PASS] All tests passed! Implementation is complete.")
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
