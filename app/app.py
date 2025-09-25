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
#
# Output:
#   - Running web server hosting the interactive dashboard
#   - WSGI server object for production deployment
# ==============================================================================

# ================= IMPORTS =================

import dash
import os
from config import EXTERNAL_STYLESHEETS, INDEX_STRING
from layout import create_app_layout
from callbacks import register_callbacks

# ================= APPLICATION INITIALIZATION =================

# Initialize the Dash web application with configuration
app = dash.Dash(
    __name__, 
    external_stylesheets=EXTERNAL_STYLESHEETS,  # Font and styling imports from config
    title="PDX Crash Risks"                     # Browser tab title
)

# Expose WSGI server for production deployment (required for hosting platforms)
server = app.server

# Apply custom HTML template with responsive design and animations
app.index_string = INDEX_STRING

# ================= LAYOUT AND CALLBACK REGISTRATION =================

# Set the complete dashboard layout structure
app.layout = create_app_layout()

# Register all interactive callback functions for user interface reactivity
register_callbacks(app)

# ================= SERVER STARTUP =================

# Run development server when executed directly (not in production)
if __name__ == '__main__':
    # Get port from environment variable 
    port = int(os.environ.get('PORT', 8050))
    # Start development server (not suitable for production use)
    app.run(host='0.0.0.0', port=port, debug=False)