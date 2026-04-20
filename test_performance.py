#!/usr/bin/env python3
"""
Quick performance test for optimized run_discovery and run_crawl functions.
"""
import time
import logging
from data_building.trade_intel.league_discovery import run_discovery
from data_building.trade_intel.trade_crawler import run_crawl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def test_discovery_performance():
    """Test the optimized discovery function with a small target."""
    print("Testing run_discovery performance...")
    start_time = time.time()
    
    # Test with a small target to avoid overwhelming the API
    discovered = run_discovery(target=50)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Discovery completed in {duration:.2f} seconds")
    print(f"Discovered {discovered} new leagues")
    print(f"Rate: {discovered/duration:.2f} leagues/second")
    return discovered, duration

def test_crawl_performance():
    """Test the optimized crawl function with a small batch."""
    print("\nTesting run_crawl performance...")
    start_time = time.time()
    
    # Test with a small batch and moderate workers
    result = run_crawl(batch_size=50, workers=15)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Crawl completed in {duration:.2f} seconds")
    print(f"Crawled {result['leagues_crawled']} leagues")
    print(f"Found {result['new_trades']} new trades")
    if result['leagues_crawled'] > 0:
        print(f"Rate: {result['leagues_crawled']/duration:.2f} leagues/second")
    return result, duration

if __name__ == "__main__":
    print("=== Performance Test for Optimized Discovery and Crawl ===")
    
    # Test discovery
    discovered, discovery_time = test_discovery_performance()
    
    # Test crawl
    crawl_result, crawl_time = test_crawl_performance()
    
    print("\n=== Summary ===")
    print(f"Discovery: {discovered} leagues in {discovery_time:.2f}s")
    print(f"Crawl: {crawl_result['leagues_crawled']} leagues, {crawl_result['new_trades']} trades in {crawl_time:.2f}s")
