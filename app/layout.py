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
from config import UI_CONFIG, UI_COLORS, UI_TYPOGRAPHY

# ================= COMPONENT FUNCTIONS =================

def create_header():
    """Create fixed header with application title and dynamic datetime display"""

    return html.Div([
        html.Div([
            # Main application title
            html.H1("Portland Crash Risk Modeling", 
                   style={
                       'fontSize': UI_TYPOGRAPHY['head'],
                       'fontWeight': '700',
                       'margin': '0',
                       'color': 'white',
                       'textAlign': 'center'
                   }),
            # Dynamic subtitle showing current selection (populated by callbacks)
            html.Div(
                "Loading...",  
                id='selected-datetime-display',
                style={
                    'fontSize': UI_TYPOGRAPHY['base'],
                    'color': UI_COLORS['text_gray'],
                    'marginTop': '0.25rem',
                    'fontWeight': '400',
                    'textAlign': 'center'
                }
            )
        ], style={
            'maxWidth': '1200px', 
            'margin': '0 auto', 
            'padding': '0 1.5rem',
            'width': '100%'
            })
    ], style={
        'background': UI_COLORS['background_dark'],
        'color': 'white',
        'height': UI_CONFIG['header_height'],
        'display': 'flex',
        'borderBottom': f"1px solid {UI_COLORS['border_gray']}",
        'alignItems': 'center',
        'flexShrink': '0',
        'zIndex': 500
    })

def create_time_slider():
    """Create fixed bottom slider for hour selection with time labels"""
    
    return html.Div([
        html.Div([
            html.Div([
                # Time display updated by callbacks
                html.Span(id='time-display', style={
                    'color': 'white',
                    'fontSize': UI_TYPOGRAPHY['lg'],
                    'fontWeight': '700'
                })
            ], style={
                'marginBottom': '0.75rem',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'minHeight': '1.5rem'  # Keep consistent height even without "Time:" label
            }),
            
            # Wrap slider + tooltip in relative container
            html.Div([
                
                # Hidden div to store hours data for clientside callback
                html.Div(id='hours-data-store', style={'display': 'none'}),

                # Custom tooltip
                html.Div(
                    id='slider-tooltip',
                    style={
                        'position': 'absolute',
                        'backgroundColor': UI_COLORS['background_dark'],
                        'color': 'white',
                        'padding': '4px 8px',
                        'borderRadius': '4px',
                        'fontSize': UI_TYPOGRAPHY['xs'],
                        'fontWeight': '600',
                        'border': f"1px solid {UI_COLORS['border_gray']}",
                        'pointerEvents': 'none',
                        'zIndex': 2000,
                        'display': 'none',
                        'whiteSpace': 'nowrap'
                    }
                ),
                
                # Slider
                dcc.Slider(
                    id='hour-slider',
                    min=0,
                    max=25,
                    step=1,
                    value=0,
                    marks={},
                    tooltip=None
                )
            ], style={'position': 'relative', 'width': '100%'})
            
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
        'position': 'fixed',
        'top': f"calc({UI_CONFIG['header_height']} + {UI_CONFIG['legend_position']['top']})",  # Add header height
        'right': UI_CONFIG['legend_position']['right'],
        'backgroundColor': UI_COLORS['background_dark'],
        'padding': UI_CONFIG['legend_padding'],          
        'borderRadius': '8px',
        'border': f"1px solid {UI_COLORS['border_gray']}",
        'color': 'white',
        'fontSize': UI_TYPOGRAPHY['sm'],                 
        'fontFamily': 'Inter',
        'zIndex': 1300,
        'minWidth': UI_CONFIG['legend_min_width']       
    }
    
    return html.Div([
        # Main interactive map component
        dash_deck.DeckGL(
        id='crash-heatmap',  # Primary map component for callbacks
        data={},  # Data populated by callbacks
        tooltip={
        'html': '<b>Risk Score:</b> {probability_text}<br><b>Street:</b> {full_name}',
            'style': {'backgroundColor': UI_COLORS['background_dark'], 'color': 'white'}
            },
        style={'position': 'absolute', 'top': '0', 'left': '0', 'right': '0', 'bottom': '0'},
        mapboxKey=""  # Uses default public key
        ),
        
        # Risk level legend overlay with gradient visualization
        html.Div([
            html.H4("Risk Score", style={'margin': '0 0 10px 0', 'fontSize': UI_TYPOGRAPHY['base']}),  
            
            # Gradient bar with numerical scale
            html.Div([
                # Gradient bar
                html.Div(style={
                    'width': UI_CONFIG['legend_gradient_width'],     
                    'height': UI_CONFIG['legend_gradient_height'],   
                    'background': 'linear-gradient(to top, rgba(26,29,35,0.31) 0%, rgba(100,100,0,0.45) 25%, rgba(200,200,0,0.59) 50%, rgba(255,130,0,0.72) 75%, rgba(255,0,0,0.86) 100%)', 
                    'borderRadius': '4px',
                    'marginRight': '10px',
                    'overflow': 'hidden',
                    'border': f"1px solid {UI_COLORS['border_gray']}"
                }),
                
                html.Div([
                    html.Div('99', style={'position': 'absolute', 'top': '-0.375rem', 'fontSize': UI_TYPOGRAPHY['xs']}),   
                    html.Div('75', style={'position': 'absolute', 'top': '2.09375rem', 'fontSize': UI_TYPOGRAPHY['xs']}), 
                    html.Div('50', style={'position': 'absolute', 'top': '4.5625rem', 'fontSize': UI_TYPOGRAPHY['xs']}),   
                    html.Div('25', style={'position': 'absolute', 'top': '7.03125rem', 'fontSize': UI_TYPOGRAPHY['xs']}),
                    html.Div('0', style={'position': 'absolute', 'top': '9.5rem', 'fontSize': UI_TYPOGRAPHY['xs']})    
                ], style={'position': 'relative', 'height': UI_CONFIG['legend_gradient_height']})  
                
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
                    'fontSize': UI_TYPOGRAPHY['base']
                }),
                # Model methodology explanation
                html.P("This map shows predicted crash risk for Portland street segments using a machine learning model trained on historical crash, weather, and road data (2007-2023). The model generates hourly predictions by combining real-time weather conditions, time patterns, street type, and historical crash patterns for each segment. Risk scores (0-100) are percentile rankings compared to predictions across all street segments and time periods. A score of 75 means the segment's predicted crash probability is higher than 75% of all predictions the model generates. Only segments above a statistically determined threshold are displayed, filtering out most low-risk segments to highlight areas with meaningfully elevated risk.", style={
                    'margin': '0 0 8px 0',
                    'fontSize': UI_TYPOGRAPHY['xs'],
                    'lineHeight': '1.4'
                }),
            
                # Data source attributions with external links
                html.Div([
                    html.Div("Weather data by ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']}),
                        html.A("Open-Meteo.com", href="https://open-meteo.com", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'}),
                    html.Br(),

                    html.Div("Crash data courtesy of ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']}),
                        html.A("ODOT Crash Reporting", href="https://tvc.odot.state.or.us/tvc/", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'}),
                        html.Div(" and ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']}),
                        html.A("Portland Metro RLIS Data", href="https://arcg.is/0CnjDC0", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'}),
                    html.Br(),

                    html.Div("Road data courtesy of ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']}),
                        html.A("PortlandMaps Open Data", href="https://gis-pdx.opendata.arcgis.com/", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'}),
                    html.Br(),

                    html.Div("Map via © ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']}),
                        html.A("Carto", href="https://carto.com/about-carto/", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'}),
                        html.Div(", © ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']}),
                        html.A("OpenStreetMap", href="https://www.openstreetmap.org/about/", target="_blank",
                            style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'}),
                        html.Div(" contributors", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs'], 'color': UI_COLORS['text_gray']})

                ]),
                html.Div([
                    html.Div("Created by Sawyer Anderson", style={'fontWeight': '600', 'fontSize': UI_TYPOGRAPHY['sm']}),
                    html.Div("Source code available on ", style={'display': 'inline', 'fontSize': UI_TYPOGRAPHY['xs']}),
                    html.A("Github", href="https://github.com/sawyerca/pdx-crash-risk", target="_blank",
                           style={'color': UI_COLORS['link_blue'], 'fontSize': UI_TYPOGRAPHY['xs'], 'textDecoration': 'underline'})  
                ], style={'margin': '8px 0 0 0'})

            ],
            id='info-tooltip-content',
            style={
                'position': 'absolute',
                'top': f"calc({UI_CONFIG['info_icon_position']['top']} + 2rem)",
                'left': '0px',
                'backgroundColor': 'rgba(15, 23, 42, 0.95)',
                'color': 'white',
                'padding': '12px',
                'borderRadius': '8px',
                'border': f"1px solid {UI_COLORS['border_gray']}",
                'fontSize': '11px',
                'fontFamily': 'Inter',
                'width': UI_CONFIG['tooltip_width'],
                'maxWidth': UI_CONFIG['tooltip_max_width'],
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
            'top': f"calc({UI_CONFIG['header_height']} + {UI_CONFIG['info_icon_position']['top']})",  # Add header height 
            'left': UI_CONFIG['info_icon_position']['left'],
            'zIndex': 1200
        }),

        # Include time control slider
        create_time_slider()
    ], style={
        'position': 'relative',
        'backgroundColor': UI_COLORS['background_darker'],
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
            'position': 'relative',  
            'zIndex': 1000,          
            'display': 'flex',
            'height': f"calc(100vh - {UI_CONFIG['header_height']})",
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