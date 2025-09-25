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

# ================= RISK LEVEL DEFINITIONS =================

# Standardized risk categories in order from lowest to highest
RISK_LEVELS = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Probability thresholds for converting continuous probabilities to categories
PROBABILITY_THRESHOLDS = {
    'very_low': 0.0,
    'low': 0.0000325,
    'medium': 0.000106,
    'high': 0.000343,
    'very_high': 0.00111
}

# Systematic opacity levels based on risk severity
OPACITY_LEVELS = {
    'very_low': 68,    # Subtle visibility for minimal risk
    'low': 68,        # Low visibility for minor risk
    'medium': 118,     # Medium visibility for moderate risk
    'high': 164,       # High visibility for significant risk
    'very_high': 230   # Maximum visibility for critical risk
}

# Base colors (RGB) for each risk level
BASE_COLORS = {
    'Very low': [26, 29, 35],       # Dark gray for minimal risk
    'Low': [127, 127, 0],         # Light yellow for low risk
    'Medium': [255, 255, 0],        # Yellow for moderate risk
    'High': [255, 127, 0],          # Orange for high risk
    'Very high': [255, 0, 0]        # Red for critical risk
}

# Complete RGBA color definitions combining base colors with opacity
RISK_COLORS = {
    level: BASE_COLORS[level] + [OPACITY_LEVELS[level.lower().replace(' ', '_')]]
    for level in RISK_LEVELS
}

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

def get_color_by_category(risk_category):
    """Get RGBA color array for categorical risk level"""

    return RISK_COLORS.get(risk_category, RISK_COLORS['Very low'])

def probability_to_category(probability):
    """Convert continuous probability to categorical risk level"""

    prob = max(0, min(1, probability))
    
    if prob >= PROBABILITY_THRESHOLDS['very_high']:
        return 'Very high'
    elif prob >= PROBABILITY_THRESHOLDS['high']:
        return 'High'
    elif prob >= PROBABILITY_THRESHOLDS['medium']:
        return 'Medium'
    elif prob >= PROBABILITY_THRESHOLDS['low']:
        return 'Low'
    else:
        return 'Very low'

def get_color_by_probability(probability, method='continuous'):
    """Get color for probability value using specified mapping method"""

    if method == 'categorical':
        # Convert probability to category first, then get color
        category = probability_to_category(probability)
        return get_color_by_category(category)
    else:
        # Use continuous interpolation between color stops
        return interpolate_color_continuous(probability)

def interpolate_color_continuous(probability):
    """Interpolate color along continuous gradient based on probability thresholds"""

    # Clamp probability to valid range
    prob = max(0, min(1, probability))
    
    # Create ordered threshold-color pairs for interpolation
    color_stops = [
        (PROBABILITY_THRESHOLDS['very_low'], RISK_COLORS['Very low']),
        (PROBABILITY_THRESHOLDS['low'], RISK_COLORS['Low']),
        (PROBABILITY_THRESHOLDS['medium'], RISK_COLORS['Medium']),
        (PROBABILITY_THRESHOLDS['high'], RISK_COLORS['High']),
        (PROBABILITY_THRESHOLDS['very_high'], RISK_COLORS['Very high'])
    ]
    
    # Find appropriate color stops for interpolation
    for i in range(len(color_stops) - 1):
        pos1, color1 = color_stops[i]
        pos2, color2 = color_stops[i + 1]
        
        if pos1 <= prob <= pos2:
            # Calculate interpolation factor between stops
            if pos2 == pos1:
                t = 0
            else:
                t = (prob - pos1) / (pos2 - pos1)
            
            # Linear interpolation for each RGBA component
            r = int(color1[0] + t * (color2[0] - color1[0]))
            g = int(color1[1] + t * (color2[1] - color1[1]))
            b = int(color1[2] + t * (color2[2] - color1[2]))
            a = int(color1[3] + t * (color2[3] - color1[3]))
            
            return [r, g, b, a]
    
    # Fallback for probabilities outside defined range
    return RISK_COLORS['Very high']

#================= CHANGE THIS WHEN BASE RATE CORRECTION IS READY =================

def get_temp_color(risk_category): # delete
    """TEMPORARY: Map categorical risk levels to RGBA colors (legacy function)"""

    return get_color_by_category(risk_category)

def get_deck_color(probability):
    """Convert crash probability to deck.gl RGBA color using continuous mapping"""

    return get_color_by_probability(probability, method='continuous')

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