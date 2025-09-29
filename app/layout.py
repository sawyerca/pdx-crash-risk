# ==============================================================================
# Dashboard Layout Components for Traffic Crash Risk Prediction
# ==============================================================================
# Purpose: Define HTML/CSS layout structure and styling for the interactive
#          crash risk prediction dashboard including header, map, controls,
#          and notification elements
# 
# Input Files:
#   - None (generates HTML/CSS structure)
#
# Output:
#   - Dash HTML components for complete dashboard layout
#   - Interactive map with legend, tooltips, and time controls
# ==============================================================================

# ================= IMPORTS =================

from dash import dcc, html
import dash_deck
from config import UI_CONFIG, UI_COLORS, get_deck_color

# ================= COMPONENT FUNCTIONS =================

def create_attribution_link(text, link_text, url):
    """Create data source attribution with clickable link"""
    
    return html.Div([
        html.Div(text, style={'display': 'inline', 'fontSize': '10px', 'color': UI_COLORS['text_gray']}),
        html.A(link_text, 
               href=url, 
               target="_blank",
               style={'color': UI_COLORS['link_blue'], 'fontSize': '10px', 'textDecoration': 'underline'}),
        html.Br()
    ])

def create_header():
    """Create fixed header with application title and dynamic datetime display"""

    return html.Div([
        html.Div([
            # Main application title
            html.H1("Portland Crash Risk Modeling", 
                   style={
                       'fontSize': '1.75rem',
                       'fontWeight': '700',
                       'margin': '0',
                       'color': 'white'
                   }),
            # Dynamic subtitle showing current selection (populated by callbacks)
            html.Div(
                "Loading...",  
                id='selected-datetime-display',
                style={
                    'fontSize': '0.875rem',
                    'color': UI_COLORS['text_gray'],
                    'marginTop': '0.25rem',
                    'fontWeight': '400'
                }
            )
        ], style={
            'maxWidth': '1200px', 
            'margin': '0 auto', 
            'padding': '0 1.5rem'
            })
    ], style={
        'background': UI_COLORS['background_dark'],
        'color': 'white',
        'height': UI_CONFIG['header_height'],
        'display': 'flex',
        'borderBottom': f"1px solid {UI_COLORS['border_gray']}",
        'alignItems': 'center',
        'flexShrink': '0'
    })

def create_time_slider():
    """Create fixed bottom slider for hour selection with time labels"""
    
    return html.Div([
        html.Div([
            # Slider header with current time display
            html.Div([
                html.Span("Time:", style={
                    'color': UI_COLORS['text_white'],
                    'fontSize': '0.875rem',
                    'fontWeight': '500',
                    'marginRight': '1rem'
                }),
                # Time display updated by callbacks
                html.Span(id='time-display', style={
                    'color': 'white',
                    'fontSize': '0.875rem',
                    'fontWeight': '600'
                })
            ], style={
                'marginBottom': '0.5rem',
                'display': 'flex',
                'alignItems': 'center'
            }),
            
            # Interactive hour selection slider
            dcc.Slider(
                id='hour-slider',
                min=0,
                max=25,
                step=1,
                value=0,
                marks={},  # Populated dynamically by callbacks
                tooltip=None
            )
        ], style={
            'padding': '1rem 2rem',
            'maxWidth': '100%'
        })
    ], style={
        'position': 'fixed',
        'bottom': UI_CONFIG['slider_position']['bottom'],
        'left': '50%',  
        'transform': 'translateX(-50%)',
        'backgroundColor': UI_COLORS['background_dark'],
        'backdropFilter': 'blur(10px)',
        'borderRadius': '8px',
        'border': f"1px solid {UI_COLORS['border_gray']}",
        'width': '90%',  
        'maxWidth': '1600px',
        'zIndex': 1000
    })

def create_map():
    """Create main map visualization with overlays, legend, and controls"""

    # Legend positioning and styling
    legend_style = {
        'position': 'absolute',
        'top': UI_CONFIG['legend_position']['top'],
        'right': UI_CONFIG['legend_position']['right'],
        'backgroundColor': UI_COLORS['background_dark'],
        'padding': '15px',
        'borderRadius': '8px',
        'border': f"1px solid {UI_COLORS['border_gray']}",
        'color': 'white',
        'fontSize': '12px',
        'fontFamily': 'Inter',
        'zIndex': 1000,
        'minWidth': '80px'
    }
    
    return html.Div([
        # Main interactive map component
        dcc.Loading(
            id="loading",
            children=[
                dash_deck.DeckGL(
                id='crash-heatmap',  # Primary map component for callbacks
                data={},  # Data populated by callbacks
                tooltip={
                'html': '<b>Risk Score:</b> {probability_text}<br><b>Street:</b> {full_name}',
                 'style': {'backgroundColor': UI_COLORS['background_dark'], 'color': 'white'}
                 },
                style={'height': '100vh', 'width': '100%', 'margin': '0 auto'},
                mapboxKey=""  # Uses default public key
            )
            ],
            type="default",
        ),
        
        # Risk level legend overlay with gradient visualization
        html.Div([
            html.H4("Risk Score", style={'margin': '0 0 10px 0', 'fontSize': '14px'}),
            
            # Gradient bar with numerical scale
            html.Div([
                # Gradient bar
                html.Div(style={
                    'width': '30px',
                    'height': '150px',
                    'background': 'linear-gradient(to top, rgba(26,29,35,0.27) 0%, rgba(127,127,0,0.27) 25%, rgba(255,255,0,0.46) 50%, rgba(255,127,0,0.64) 75%, rgba(255,0,0,0.9) 100%)', 
                    'borderRadius': '4px',
                    'marginRight': '10px',
                    'overflow': 'hidden'
                }),
                
                # Numerical labels
                html.Div([
                    html.Div('99', style={'position': 'absolute', 'top': '-5px', 'fontSize': '11px'}),
                    html.Div('75', style={'position': 'absolute', 'top': '32.5px', 'fontSize': '11px'}),
                    html.Div('50', style={'position': 'absolute', 'top': '70px', 'fontSize': '11px'}),
                    html.Div('25', style={'position': 'absolute', 'top': '107.5px', 'fontSize': '11px'}),
                    html.Div('0', style={'position': 'absolute', 'top': '145px', 'fontSize': '11px'})  
                ], style={'position': 'relative', 'height': '150px'})
                
            ], style={'display': 'flex', 'alignItems': 'flex-start'})
        ], style=legend_style),  

        # Notification for data updates (hidden by default)
        html.Div(
            "Refresh to see new predictions",
            id='refresh-notification',
            style={'display': 'none'}  # Shown/hidden by callbacks
        ),
        
        # Hidden tracking components for callback functionality
        html.Div(id='page-load-tracker', style={'display': 'none'}),
        dcc.Interval(
            id='refresh-check-interval',
            interval=30000,
            n_intervals=0
        ),

        # Information tooltip with model and data source details
        html.Div([
            # Clickable info icon
            html.Div("ℹ️", 
                id='info-icon',
                style={
                    'fontSize': '30px',
                    'cursor': 'pointer',
                    'opacity': '1',
                    'transition': 'opacity 0.2s ease'
                }
            ),
            
            # Detailed tooltip content (visible on hover)
            html.Div([
                html.P("Map info:", style={
                    'margin': '0 0 8px 0',
                    'fontWeight': '600',
                    'fontSize': '13px'
                }),
                # Model methodology explanation
                html.P("This map displays predicted crash risk for Portland street segments using an XGBoost model trained on historical crash and weather data (2019-2023). Each colored line represents a street segment, with colors indicating percentile-based risk scores (0-100) from very low (dark gray) to very high (red) for the selected hour. The model predicts hourly crash occurrence probability by incorporating real-time weather conditions, temporal patterns (time of day, day of week), street characteristics, and historical crash patterns for each segment. Only segments with elevated risk above a calculated threshold (knee point) are displayed. Risk scores represent percentile rankings within this filtered population of higher-risk segments.", style={
                    'margin': '0 0 8px 0',
                    'fontSize': '11px',
                    'lineHeight': '1.4'
                }),
            
                # Data source attributions with external links
                html.Div([
                    create_attribution_link("Weather data by ", "Open-Meteo.com", "https://open-meteo.com"),
                    create_attribution_link("Crash data courtesy of ", "ODOT Crash Reporting", "https://tvc.odot.state.or.us/tvc/"),
                    create_attribution_link("Road data courtesy of ", "PortlandMaps Open Data", "https://gis-pdx.opendata.arcgis.com/"),
                        html.Div("Map via © ", style={'display': 'inline', 'fontSize': '10px', 'color': UI_COLORS['text_gray']}),
                        html.A("Carto", href="https://carto.com/about-carto/", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': '10px', 'textDecoration': 'underline'}),
                        html.Div(", © ", style={'display': 'inline', 'fontSize': '10px', 'color': UI_COLORS['text_gray']}),
                        html.A("OpenStreetMap", href="http://www.openstreetmap.org/about/", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': '10px', 'textDecoration': 'underline'}),
                        html.Div(" contributors", style={'display': 'inline', 'fontSize': '10px', 'color': UI_COLORS['text_gray']})

                ])

            ],
            id='info-tooltip-content',
            style={
                'position': 'absolute',
                'top': '30px',
                'left': '0px',
                'backgroundColor': 'rgba(15, 23, 42, 0.95)',
                'color': 'white',
                'padding': '12px',
                'borderRadius': '8px',
                'border': f"1px solid {UI_COLORS['border_gray']}",
                'fontSize': '11px',
                'fontFamily': 'Inter',
                'width': UI_CONFIG['tooltip_width'],
                'backdropFilter': 'blur(10px)',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
                'opacity': '0',
                'visibility': 'hidden',
                'transition': 'opacity 0.3s ease, visibility 0.3s ease',
                'pointerEvents': 'none'
            })
        ], 
        style={
            'position': 'fixed',
            'top': UI_CONFIG['info_icon_position']['top'],
            'left': UI_CONFIG['info_icon_position']['left'],
            'zIndex': 1200
        }),

        # Include time control slider
        create_time_slider()
    ], style={
        'flex': '1',
        'backgroundColor': UI_COLORS['background_darker'],
        'height': '100%',
        'width': '100%',
        'overflow' : 'hidden'
    })

def create_app_layout():
    """Assemble complete dashboard layout with header and main content area"""

    return html.Div([
        create_header(),
        # Main content container (currently map only, expandable for sidebar)
        html.Div([
            create_map()
        ], style={
            'display': 'flex',
            'height': f"calc(100vh - {UI_CONFIG['header_height']})",
            'overflow': 'hidden',
            'width': '100vw'
        })
    ], style={
        'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        'backgroundColor': UI_COLORS['background_darker'],
        'color': 'white',
        'height': '100vh',
        'width': '100vw',
        'margin': '0',
        'padding': '0',
        'overflow': 'hidden'
    })