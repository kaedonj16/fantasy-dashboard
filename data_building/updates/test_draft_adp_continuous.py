#!/usr/bin/env python3
"""
Test script for the continuous draft ADP crawler functionality.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_building.trade_intel.draft_adp_crawler import run_draft_adp_crawl_continuous
import logging

# Set up logging to see the output
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def test_continuous_mode():
    """Test the continuous crawling mode with a very short duration."""
    print("Testing continuous draft ADP crawler...")
    print("Running for 0.1 hours (6 minutes) with 2-minute intervals...")
    
    result = run_draft_adp_crawl_continuous(
        batch_size=100,  # Small batch for testing
        workers=2,        # Fewer workers for testing
        interval_minutes=2,  # 2-minute intervals
        hours=0.1,        # 6 minutes total
    )
    
    print(f"Test completed! Result: {result}")
    return result

if __name__ == "__main__":
    test_continuous_mode()
