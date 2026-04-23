#!/usr/bin/env python3
"""
Run trade intelligence discovery and crawl for multiple iterations.
This will run continuously for several hours, discovering new leagues
and crawling trade data from them.
"""

import time
import logging
from datetime import datetime, timedelta

from data_building.trade_intel.league_discovery import run_discovery
from data_building.trade_intel.trade_crawler import run_crawl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("trade_intel_iterations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_iterations(duration_hours: int = 4, min_sleep_minutes: int = 5):
    """
    Run discovery and crawl iterations for specified duration.
    
    Args:
        duration_hours: How long to run (default 4 hours)
        min_sleep_minutes: Minimum sleep between iterations (default 5 minutes)
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    iteration = 0
    
    logger.info(f"Starting trade intelligence iterations for {duration_hours} hours")
    logger.info(f"Will run until {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    while datetime.now() < end_time:
        iteration += 1
        iteration_start = datetime.now()
        
        logger.info(f"\n=== Iteration {iteration} at {iteration_start.strftime('%H:%M:%S')} ===")
        
        try:
            # Run discovery
            logger.info("Running league discovery...")
            n = run_discovery()
            logger.info(f"Discovered {n} new leagues")
            
            if n > 0:
                # Run crawl with discovered leagues
                logger.info(f"Running crawl with batch_size={n}...")
                result = run_crawl(batch_size=n)
                logger.info(f"Crawl result: {result}")
            else:
                logger.info("No new leagues discovered, skipping crawl")
                
        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {e}")
            
        iteration_duration = (datetime.now() - iteration_start).total_seconds()
        logger.info(f"Iteration {iteration} completed in {iteration_duration:.1f} seconds")
        
        # Sleep between iterations, but don't overshoot the end time
        time_remaining = (end_time - datetime.now()).total_seconds()
        sleep_time = min(min_sleep_minutes * 60, time_remaining - 60)  # Leave 1 minute buffer
        
        if sleep_time > 0:
            logger.info(f"Sleeping for {sleep_time/60:.1f} minutes...")
            time.sleep(sleep_time)
        else:
            logger.info("Near end time, starting next iteration immediately")
    
    total_duration = (datetime.now() - start_time).total_seconds() / 3600
    logger.info(f"\nCompleted {iteration} iterations in {total_duration:.2f} hours")

if __name__ == "__main__":
    # Run for 4 hours by default, or override with command line args
    import sys
    hours = 4
    if len(sys.argv) > 1:
        hours = int(sys.argv[1])
    
    run_iterations(duration_hours=hours)
