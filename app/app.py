# ==============================================================================
# Main Application Entry Point for Traffic Crash Risk Prediction Dashboard
# ==============================================================================
# Purpose: Initialize and configure the Dash web application for the Portland
#          crash risk prediction system, register callbacks, and start the server
# 
# Input Files:
#   - config.py: Application configuration and styling constants
#   - layout.py: Dashboard HTML/CSS component definitions
#   - callbacks.py: Interactive callback function registration
#   - stats.py: Visitor statistics endpoint
#
# Output:
#   - Running web server hosting the interactive dashboard
#   - WSGI server object for production deployment
# ==============================================================================

# ================= ENVIRONMENT SETUP =================

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# ================= IMPORTS =================

import dash
import os
from datetime import datetime
import pytz
from config import EXTERNAL_STYLESHEETS, INDEX_STRING
from layout import create_app_layout
from callbacks import register_callbacks
from stats import register_stats_endpoint

# ================= APPLICATION INITIALIZATION =================

# Initialize the Dash web application with configuration
app = dash.Dash(
    __name__, 
    external_stylesheets=EXTERNAL_STYLESHEETS,  # Font and styling imports from config
    title="PDX Crash Risk"                     # Browser tab title
)

# Expose WSGI server for production deployment (required for hosting platforms)
server = app.server

# Configure Flask session for visitor tracking
server.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Apply custom HTML template with responsive design and animations
app.index_string = INDEX_STRING

# ================= VISITOR TRACKING =================

# In-memory storage for visitor statistics
visitor_stats = {
    'unique_sessions': set(),
    'visit_timestamps': [],
    'start_time': datetime.now(pytz.timezone('America/Los_Angeles'))
}

# ================= LAYOUT AND CALLBACK REGISTRATION =================

# Set the complete dashboard layout structure
app.layout = create_app_layout()

# Register all interactive callback functions for user interface reactivity
register_callbacks(app, visitor_stats)

# Register the stats endpoint
register_stats_endpoint(server, visitor_stats)

# ================= SERVER TESTING =================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)