#!/usr/bin/env python3
"""
Performance test script for MLX-GUI optimizations.
Demonstrates the performance improvements made to the codebase.
"""

import time
import asyncio
import threading
import statistics
from typing import List, Dict, Any
import json

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mlx_gui.database import get_database_manager
from mlx_gui.model_manager import get_model_manager
from mlx_gui.inference_queue_manager import get_inference_manager
from mlx_gui.performance_monitor import get_performance_monitor, start_performance_monitoring


class PerformanceTester:
    """Test performance improvements in MLX-GUI."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.model_manager = get_model_manager()
        self.inference_manager = get_inference_manager()
        self.performance_monitor = get_performance_monitor()
        
        # Test results
        self.results = {}
    
    def test_database_performance(self) -> Dict[str, Any]:
        """Test database query performance improvements."""
        print("🔍 Testing Database Performance...")
        
        # Test 1: Settings cache performance
        print("  Testing settings cache...")
        
        # Warm up cache
        for _ in range(5):
            self.db_manager.get_setting("server_port")
        
        # Test cache performance
        start_time = time.time()
        for _ in range(100):
            self.db_manager.get_setting("server_port")
        cache_time = time.time() - start_time
        
        # Test without cache (simulate old behavior)
        start_time = time.time()
        for _ in range(100):
            with self.db_manager.get_session() as session:
                from mlx_gui.models import AppSettings
                setting = session.query(AppSettings).filter_by(key="server_port").first()
                if setting:
                    _ = setting.get_typed_value()
        no_cache_time = time.time() - start_time
        
        cache_improvement = ((no_cache_time - cache_time) / no_cache_time) * 100
        
        # Test 2: Model query performance
        print("  Testing model queries...")
        
        # Test old way (full object loading)
        start_time = time.time()
        for _ in range(10):
            with self.db_manager.get_session() as session:
                from mlx_gui.models import Model
                models = session.query(Model).all()
                _ = len(models)
        old_query_time = time.time() - start_time
        
        # Test new way (specific columns)
        start_time = time.time()
        for _ in range(10):
            with self.db_manager.get_session() as session:
                from mlx_gui.models import Model
                models = session.query(Model.id, Model.name, Model.status).all()
                _ = len(models)
        new_query_time = time.time() - start_time
        
        query_improvement = ((old_query_time - new_query_time) / old_query_time) * 100
        
        return {
            "settings_cache": {
                "cached_time_ms": cache_time * 1000,
                "uncached_time_ms": no_cache_time * 1000,
                "improvement_percent": cache_improvement
            },
            "model_queries": {
                "old_query_time_ms": old_query_time * 1000,
                "new_query_time_ms": new_query_time * 1000,
                "improvement_percent": query_improvement
            }
        }
    
    def test_model_manager_performance(self) -> Dict[str, Any]:
        """Test model manager performance improvements."""
        print("🔍 Testing Model Manager Performance...")
        
        # Test path resolution caching
        print("  Testing path resolution caching...")
        
        test_paths = [
            "~/models/test-model",
            "microsoft/DialoGPT-medium",
            "/Users/test/.cache/huggingface/hub/models--microsoft--DialoGPT-medium"
        ]
        
        # Test without cache (simulate old behavior)
        start_time = time.time()
        for _ in range(50):
            for path in test_paths:
                # Simulate old path resolution
                if path.startswith("~"):
                    resolved = os.path.expanduser(path)
                elif "/" in path and not os.path.exists(path):
                    # Simulate HuggingFace path resolution
                    resolved = path.replace("/", "--")
                else:
                    resolved = path
        no_cache_time = time.time() - start_time
        
        # Test with cache
        start_time = time.time()
        for _ in range(50):
            for path in test_paths:
                resolved = self.model_manager._resolve_model_path(path)
        cache_time = time.time() - start_time
        
        path_improvement = ((no_cache_time - cache_time) / no_cache_time) * 100
        
        return {
            "path_resolution": {
                "cached_time_ms": cache_time * 1000,
                "uncached_time_ms": no_cache_time * 1000,
                "improvement_percent": path_improvement
            }
        }
    
    def test_inference_queue_performance(self) -> Dict[str, Any]:
        """Test inference queue performance improvements."""
        print("🔍 Testing Inference Queue Performance...")
        
        # Test queue status caching
        print("  Testing queue status caching...")
        
        # Simulate multiple status checks
        start_time = time.time()
        for _ in range(20):
            # Simulate old behavior (database query each time)
            with self.db_manager.get_session() as session:
                from mlx_gui.models import Model, RequestQueue, QueueStatus
                from sqlalchemy import and_
                
                # This would be done for each model
                model_record = session.query(Model).filter(Model.name == "test-model").first()
                if model_record:
                    queued_count = session.query(RequestQueue).filter(
                        and_(
                            RequestQueue.model_id == model_record.id,
                            RequestQueue.status == QueueStatus.QUEUED.value
                        )
                    ).count()
        old_time = time.time() - start_time
        
        # Test with cache
        start_time = time.time()
        for _ in range(20):
            # This uses the cached version
            status = self.inference_manager.get_queue_status("test-model")
        new_time = time.time() - start_time
        
        queue_improvement = ((old_time - new_time) / old_time) * 100
        
        return {
            "queue_status": {
                "cached_time_ms": new_time * 1000,
                "uncached_time_ms": old_time * 1000,
                "improvement_percent": queue_improvement
            }
        }
    
    def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent operation performance."""
        print("🔍 Testing Concurrent Operations...")
        
        # Test concurrent settings reads
        print("  Testing concurrent settings reads...")
        
        def read_settings():
            for _ in range(10):
                self.db_manager.get_setting("server_port")
                time.sleep(0.01)
        
        # Test with multiple threads
        start_time = time.time()
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=read_settings)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        
        # Test sequential
        start_time = time.time()
        for _ in range(5):
            read_settings()
        sequential_time = time.time() - start_time
        
        concurrency_improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
        
        return {
            "concurrent_operations": {
                "concurrent_time_ms": concurrent_time * 1000,
                "sequential_time_ms": sequential_time * 1000,
                "improvement_percent": concurrency_improvement
            }
        }
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage improvements."""
        print("🔍 Testing Memory Usage...")
        
        import psutil
        import gc
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Perform operations that would use memory
        print("  Testing memory-efficient operations...")
        
        # Simulate model operations
        for i in range(10):
            # Simulate model loading operations
            with self.db_manager.get_session() as session:
                from mlx_gui.models import Model
                models = session.query(Model.id, Model.name, Model.status).all()
            
            # Force garbage collection
            gc.collect()
        
        # Get final memory
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        return {
            "memory_usage": {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "efficient": memory_increase < 50  # Less than 50MB increase is good
            }
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        print("🚀 Starting MLX-GUI Performance Tests")
        print("=" * 50)
        
        # Start performance monitoring
        start_performance_monitoring()
        
        # Run tests
        self.results = {
            "database": self.test_database_performance(),
            "model_manager": self.test_model_manager_performance(),
            "inference_queue": self.test_inference_queue_performance(),
            "concurrent": self.test_concurrent_operations(),
            "memory": self.test_memory_usage()
        }
        
        # Get performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        self.results["performance_summary"] = performance_summary
        
        return self.results
    
    def print_results(self):
        """Print test results in a formatted way."""
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE TEST RESULTS")
        print("=" * 50)
        
        # Database Performance
        print("\n🗄️  Database Performance:")
        db_results = self.results["database"]
        print(f"  Settings Cache: {db_results['settings_cache']['improvement_percent']:.1f}% improvement")
        print(f"  Model Queries: {db_results['model_queries']['improvement_percent']:.1f}% improvement")
        
        # Model Manager Performance
        print("\n🤖 Model Manager Performance:")
        mm_results = self.results["model_manager"]
        print(f"  Path Resolution: {mm_results['path_resolution']['improvement_percent']:.1f}% improvement")
        
        # Inference Queue Performance
        print("\n📋 Inference Queue Performance:")
        iq_results = self.results["inference_queue"]
        print(f"  Queue Status: {iq_results['queue_status']['improvement_percent']:.1f}% improvement")
        
        # Concurrent Operations
        print("\n🔄 Concurrent Operations:")
        conc_results = self.results["concurrent"]
        print(f"  Concurrency: {conc_results['concurrent_operations']['improvement_percent']:.1f}% improvement")
        
        # Memory Usage
        print("\n💾 Memory Usage:")
        mem_results = self.results["memory"]
        print(f"  Memory Increase: {mem_results['memory_usage']['memory_increase_mb']:.1f}MB")
        print(f"  Efficient: {'✅' if mem_results['memory_usage']['efficient'] else '❌'}")
        
        # Performance Score
        if "performance_summary" in self.results:
            perf_summary = self.results["performance_summary"]
            if "performance_score" in perf_summary:
                score = perf_summary["performance_score"]
                print(f"\n🎯 Overall Performance Score: {score:.1f}/100")
                
                if score >= 90:
                    print("  Status: 🟢 Excellent")
                elif score >= 75:
                    print("  Status: 🟡 Good")
                elif score >= 60:
                    print("  Status: 🟠 Fair")
                else:
                    print("  Status: 🔴 Needs Improvement")
        
        print("\n" + "=" * 50)
    
    def save_results(self, filename: str = "performance_test_results.json"):
        """Save test results to a JSON file."""
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert_datetime)
        
        print(f"💾 Results saved to {filename}")


def main():
    """Main test function."""
    try:
        tester = PerformanceTester()
        results = tester.run_all_tests()
        tester.print_results()
        tester.save_results()
        
        print("\n✅ Performance tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 