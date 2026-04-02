#!/usr/bin/env python3
"""
Production Startup Script
Entry point for Render deployment that handles initialization
before starting the Flask application.
"""

import os
import sys
from datetime import datetime

def main():
    print("🌟 Production Startup - Fantasy Dashboard")
    print(f"📅 Started: {datetime.now().isoformat()}")
    
    # Check if this is the first run
    first_run_flag = "/tmp/fantasy_dashboard_initialized"
    
    if not os.path.exists(first_run_flag):
        print("🔧 First deployment detected - running initialization...")
        
        # Run the comprehensive initialization
        try:
            from scripts.initialize_production import main as init_main
            init_main()
            
            # Create flag file to prevent re-initialization
            with open(first_run_flag, 'w') as f:
                f.write(f"Initialized: {datetime.now().isoformat()}")
            
            print("✅ First-time initialization completed successfully")
            
        except Exception as e:
            print(f"❌ First-time initialization failed: {e}")
            print("🔄 Continuing with app startup (manual initialization may be needed)")
    
    else:
        print("🔄 Existing deployment detected - skipping initialization")
        with open(first_run_flag, 'r') as f:
            init_time = f.read().strip()
            print(f"📅 Previously initialized: {init_time}")
    
    # Start the Flask application
    print("\n" + "="*60)
    print("🚀 STARTING FLASK APPLICATION")
    print("="*60)
    
    try:
        # Import and run the Flask app
        from app import app
        
        # Get port from environment (Render provides this)
        port = int(os.environ.get('PORT', 5000))
        
        print(f"🌐 Server starting on port {port}")
        print(f"🌍 Environment: {os.environ.get('PYTHON_ENV', 'development')}")
        print(f"🔗 Database URL: {'✅ Set' if os.environ.get('DATABASE_URL') else '❌ Not set'}")
        
        # Start the application
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Failed to start Flask application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
