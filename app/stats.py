# ==============================================================================
# Statistics Dashboard Endpoint
# ==============================================================================
# Purpose: Provide protected endpoint for viewing visitor statistics
# 
# Input:
#   - Flask server instance
#   - visitor_stats dictionary with tracking data
#
# Output:
#   - Formatted HTML page with visitor metrics at /stats endpoint
# ==============================================================================

# ================= IMPORTS =================

import os
from datetime import datetime
import pytz
from flask import request

# ================= STATS ENDPOINT =================

def register_stats_endpoint(server, visitor_stats):
    """Register the /stats endpoint with the Flask server"""
    
    @server.route('/stats')
    def stats():
        """Protected endpoint to view visitor statistics"""
        
        # Check password
        password = request.args.get('password')
        correct_password = os.environ.get('STATS_PASSWORD')
        
        if not correct_password:
            return """
            <html>
                <body style="font-family: Inter, sans-serif; background: #0f172a; color: white; padding: 40px;">
                    <h1>Stats Not Configured</h1>
                    <p>STATS_PASSWORD environment variable is not set.</p>
                    <p>Please set it on your deployment platform.</p>
                </body>
            </html>
            """, 500
        
        if password != correct_password:
            return """
            <html>
                <body style="font-family: Inter, sans-serif; background: #0f172a; color: white; padding: 40px;">
                    <h1>Access Denied</h1>
                    <p>Invalid password. Please check your URL.</p>
                </body>
            </html>
            """, 403
        
        # Calculate statistics
        unique_visitors = len(visitor_stats['unique_sessions'])
        total_visits = len(visitor_stats['visit_timestamps'])
        
        start_time = visitor_stats['start_time']
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        uptime = current_time - start_time
        
        # Calculate uptime components
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        # Format uptime string
        uptime_parts = []
        if days > 0:
            uptime_parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            uptime_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0 or not uptime_parts:
            uptime_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        uptime_str = ", ".join(uptime_parts)
        
        # Calculate visits per day 
        days_running = max(uptime.total_seconds() / 86400, 0.1)
        
        # Return formatted HTML page
        return f"""
        <html>
            <head>
                <title>PDX Crash Risk - Stats</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            </head>
            <body style="font-family: Inter, sans-serif; background: #0f172a; color: white; padding: 40px; margin: 0;">
                <div style="max-width: 800px; margin: 0 auto;">
                    <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">Dashboard Statistics</h1>
                    <p style="color: #cbd5e1; margin-bottom: 2rem;">Portland Crash Risk Modeling</p>
                    
                    <div style="background: #1e293b; border: 1px solid #475569; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                        <h2 style="font-size: 1.25rem; margin-top: 0; margin-bottom: 1rem; color: #60a5fa;">Visitor Metrics</h2>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <div style="font-size: 2.5rem; font-weight: 700; color: #60a5fa;">{unique_visitors}</div>
                                <div style="color: #cbd5e1; font-size: 0.875rem;">Unique Visitors</div>
                            </div>
                            <div>
                                <div style="font-size: 2.5rem; font-weight: 700; color: #34d399;">{total_visits}</div>
                                <div style="color: #cbd5e1; font-size: 0.875rem;">Total Page Views</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: #1e293b; border: 1px solid #475569; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                        <h2 style="font-size: 1.25rem; margin-top: 0; margin-bottom: 1rem; color: #60a5fa;">Uptime</h2>
                        <div style="font-size: 1.25rem; margin-bottom: 0.5rem;">{uptime_str}</div>
                        <div style="color: #cbd5e1; font-size: 0.875rem;">Started: {start_time.strftime('%B %d, %Y at %I:%M %p %Z')}</div>
                    </div>
                </div>
            </body>
        </html>
        """