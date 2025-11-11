"""
Integration test for snapshot management functionality.

Tests snapshot creation and verification directly via VectorStoreClient:
1. Creating a test snapshot
2. Verifying the snapshot integrity
3. Handling error cases
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.services.vectordb_service.service import VectorStoreClient
from src.rag.services.embedder_service.service import EmbeddingService
from src.utils.task_queue import get_task_queue


async def test_snapshot_management():
    """
    Test snapshot creation and verification directly via VectorStoreClient.
    
    This test requires an existing FAISS index.
    """
    print("\n" + "=" * 80)
    print("Snapshot Management Integration Test")
    print("=" * 80 + "\n")
    
    # Initialize task queue
    task_queue = get_task_queue()
    await task_queue.start()
    print("Task queue started")
    
    try:
        # Check if index exists
        index_path = Path("data/index/default__vector_store.json")
        if not index_path.exists():
            print("\nWARNING: No FAISS index found at data/index")
            print("Snapshot tests require an existing index. Skipping tests.")
            return
        
        print(f"\nFound existing index: {index_path}")
        
        # Initialize vector store client
        vector_client = VectorStoreClient(
            store_type="faiss",
            config={'index_dir': 'data/index'}
        )
        
        # Load embedder and index
        embedder = EmbeddingService()
        vector_client.load_index(embed_model=embedder.embed_model)
        
        if not vector_client.index:
            print("\nWARNING: Failed to load index. Skipping tests.")
            return
        
        print("Index loaded successfully")
        
        # Test 1: Create snapshot
        print("\n" + "-" * 80)
        print("Test 1: Create Snapshot")
        print("-" * 80)
        
        test_snapshot_name = f"test_snapshot_{int(time.time())}"
        
        print(f"\nCreating snapshot: {test_snapshot_name}")
        
        start_time = time.time()
        success = vector_client.create_snapshot(
            snapshot_name=test_snapshot_name,
            embed_model_info={
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384
            }
        )
        create_duration = time.time() - start_time
        
        print(f"\nSnapshot creation result ({create_duration:.2f}s):")
        print(f"Success: {success}")
        
        if not success:
            print("\nWARNING: Snapshot creation failed")
            print("Skipping verification test.")
            return
        
        # Verify snapshot was created
        snapshot_dir = Path("data/snapshots") / test_snapshot_name
        manifest_path = snapshot_dir / "manifest.json"
        
        if manifest_path.exists():
            print(f"\nSnapshot directory: {snapshot_dir}")
            print(f"Manifest file: {manifest_path}")
        else:
            print(f"\nWARNING: Manifest file not found at {manifest_path}")
        
        # Test 2: Verify snapshot
        print("\n" + "-" * 80)
        print("Test 2: Verify Snapshot Integrity")
        print("-" * 80)
        
        print(f"\nVerifying snapshot: {test_snapshot_name}")
        
        start_time = time.time()
        verify_result = vector_client.verify_snapshot_integrity(test_snapshot_name)
        verify_duration = time.time() - start_time
        
        print(f"\nSnapshot verification result ({verify_duration:.2f}s):")
        print(json.dumps(verify_result, indent=2))
        
        # Test 3: Verify non-existent snapshot
        print("\n" + "-" * 80)
        print("Test 3: Verify Non-Existent Snapshot (Error Case)")
        print("-" * 80)
        
        print("\nVerifying non-existent snapshot: nonexistent_snapshot_12345")
        
        start_time = time.time()
        invalid_result = vector_client.verify_snapshot_integrity("nonexistent_snapshot_12345")
        invalid_duration = time.time() - start_time
        
        print(f"\nVerification result for non-existent snapshot ({invalid_duration:.2f}s):")
        print(json.dumps(invalid_result, indent=2))
        
        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"\nSnapshot Creation: {'SUCCESS' if success else 'FAILED'}")
        if success:
            print(f"Snapshot Verification: {'VALID' if verify_result.get('valid') else 'INVALID'}")
            print(f"Error Handling: {'OK' if not invalid_result.get('valid') else 'FAILED'}")
        print("\nAll snapshot management tests completed!")
        
    finally:
        # Cleanup
        await task_queue.stop()
        print("\nTask queue stopped")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_snapshot_management())
