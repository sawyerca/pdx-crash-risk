# ==============================================================================
# Configuration Constants for Traffic Crash Risk Prediction Dashboard
# ==============================================================================
# Purpose: Centralized configuration for map visualization, color schemes,
#          styling constants, and dashboard appearance settings
# 
# Input Files:
#   - None (defines constants and utility functions)
#
# Output:
#   - Color mapping functions for risk visualization
#   - Map configuration parameters
#   - Dashboard styling constants
# ==============================================================================

# ================= MAP CONFIGURATION =================

# Map display and interaction settings
MAP_CONFIG = {
    'center': {'lat': 45.43350, 'lon': -122.70312},  # Portland, Oregon coordinates
    'initial_zoom': 10,                                # City-wide view
    'min_zoom': 8,                                     # Minimum zoom level
    'max_zoom': 16,                                    # Maximum zoom level
    'map_style': 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    'path_width': 10,                                  # Default street segment width
    'min_path_width': 2,                              # Minimum path width in pixels
    'highlight_color': [255, 255, 255, 100]          # Color for highlighted segments
}

# Backward compatibility for existing code
MAP_CENTER = MAP_CONFIG['center']
MAP_ZOOM = MAP_CONFIG['initial_zoom']

# ================= PERFORMANCE CONFIGURATION =================

# Performance and caching settings
PERFORMANCE_CONFIG = {
    'max_segments_display': 15000,        # Maximum segments to render simultaneously
    'update_interval_seconds': 30,        # Background update frequency
    'cache_timeout_hours': 2,             # Data cache expiration time
    'memory_warning_threshold': 85.0,     # RAM usage percentage warning level
    'chunk_size_hours': 6,                # Default processing chunk size
    'pregeneration_batch_size': 5         # Maps to generate per batch
}

# ================= DASHBOARD STYLING =================

# UI layout and positioning constants
UI_CONFIG = {
    'header_height': '70px',
    'sidebar_width': '350px',              # For future sidebar implementation
    'legend_position': {'top': '20px', 'right': '20px'},
    'info_icon_position': {'top': '80px', 'left': '10px'},
    'slider_position': {'bottom': '20px'},
    'notification_position': {'bottom': '150px', 'right': '20px'},
    'tooltip_width': '500px'
}

# Color scheme for UI elements
UI_COLORS = {
    'background_dark': '#0f172a',          # Primary dark background
    'background_darker': '#0e0e0e',        # Secondary darker background
    'border_gray': '#475569',              # Border and accent color
    'text_white': '#ffffff',               # Primary text color
    'text_gray': '#cbd5e1',                # Secondary text color
    'link_blue': '#60a5fa'                 # Link and accent color
}

# ================= COLOR MAPPING FUNCTIONS =================

def get_deck_color(risk_score):
    """Get color for display based on risk score with interpolation"""
    
    # Define color stops with score thresholds
    color_stops = [
        (0,   [26, 29, 35],   68),   # Very low
        (25,  [127, 127, 0],  68),   # Low  
        (50,  [255, 255, 0],  118),  # Medium
        (75,  [255, 127, 0],  164),  # High
        (100, [255, 0, 0],    230)   # Very high
    ]
    
    # Find the two stops to interpolate between
    for i in range(len(color_stops) - 1):
        score1, color1, opacity1 = color_stops[i]
        score2, color2, opacity2 = color_stops[i + 1]
        
        if score1 <= risk_score <= score2:
            # Calculate interpolation factor (0 = first stop, 1 = second stop)
            if score2 == score1:
                t = 0  # Avoid division by zero
            else:
                t = (risk_score - score1) / (score2 - score1)
            
            # Interpolate RGB values
            r = int(color1[0] + t * (color2[0] - color1[0]))
            g = int(color1[1] + t * (color2[1] - color1[1]))
            b = int(color1[2] + t * (color2[2] - color1[2]))
            
            # Interpolate opacity
            a = int(opacity1 + t * (opacity2 - opacity1))
            
            return [r, g, b, a]
    
    # Fallback to very low color
    return [26, 29, 35, 68] 

# ================= EXTERNAL RESOURCES =================

# External stylesheet URLs for typography and styling
EXTERNAL_STYLESHEETS = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
]

# Custom HTML template with responsive design and interactive features
INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Global reset for full-viewport layout */
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
            }
            
            /* Smooth fade-in animation for notifications */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Interactive tooltip hover states */
            #info-icon:hover {
                opacity: 1 !important;
            }
            
            /* Keep tooltip visible when hovering over either icon or content */
            #info-icon:hover + #info-tooltip-content,
            #info-tooltip-content:hover {
                opacity: 1 !important;
                visibility: visible !important;
                pointer-events: auto !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''