# ==============================================================================
# Street Network Segmentation and Weather Station Assignment
# ==============================================================================
# Purpose: Process street centerline data into uniform ~300ft segments with
#          nearest weather station assignments for crash prediction modeling
# 
# Input Files:
#   - streets.geojson: Street centerline network data
#   - ../data/id_lookup.csv: Weather station locations
#   - crashes.csv: Traffic crash records for geographic bounds
#
# Output:
#   - ../data/street_seg.parquet: Segmented street network with weather linkages
# ==============================================================================

# ================= LIBRARIES =================

library(tidyverse)
library(sf)
library(janitor)
library(data.table)
library(RANN)

# ================= LOAD AND CLASSIFY ROAD DATA =================

# Load streets and simplify road type classifications
streets <- st_read("streets.geojson") %>%
  clean_names() %>%
  select(full_name, type, geometry, length) %>%
  mutate(
    type = case_when(
      type %in% c(1110, 1120, 1121, 1122, 1123, 5101) ~ "1_fwy",  
      type %in% c(1200, 1221, 1222, 1223, 5201) ~ "2_hwy",           
      type %in% c(1300, 1321, 5301) ~ "3_art1",                    
      type %in% c(1400, 1421, 5401, 5402) ~ "4_art2",               
      type %in% c(1450, 1471, 5450, 5451) ~ "5_art3",              
      type %in% c(1500, 1521, 5500, 5501) ~ "6_res",               
      type %in% c(1700, 1780, 1800, 1950, 7700, 8224, 9000) ~ "7_oth" 
    )
  )

# ================= LOAD REFERENCE DATA =================

# Load weather station locations for nearest neighbor assignment
id_lookup <- fread("../data/id_lookup.csv", encoding = "UTF-8") %>% 
  clean_names()

# Load crash data to define geographic study area bounds
crashes <- fread("crashes.csv", encoding = "UTF-8") %>% clean_names()

# ================= SEGMENT ALL STREETS =================

# Segments long streets into uniform pieces less than 300 ft
segment_linestring <- function(linestring, road_length, max_length = 300, road_attributes) {
  
  # Keep short roads intact
  if (road_length <= max_length) {
    result <- road_attributes
    result$segment_length <- road_length
    result$geometry <- st_sfc(linestring, crs = st_crs(linestring))
    return(result)
  }
  
  # Calculate equal-length segments
  n_segments <- ceiling(road_length / max_length)
  segment_length <- road_length / n_segments
  
  # Extract geometric segments along the linestring
  segments_list <- list()
  geometries_list <- list()
  
  for (i in 1:n_segments) {
    start_distance <- (i - 1) * segment_length
    end_distance <- i * segment_length
    
    # Create segment geometry using proportional distances along line
    segment_geom <- st_linesubstring(linestring, 
                                     start_distance / road_length, 
                                     end_distance / road_length)
    
    # Preserve road attributes for each segment
    segment_data <- road_attributes
    segment_data$segment_length <- segment_length
    
    segments_list[[i]] <- segment_data
    geometries_list[[i]] <- segment_geom
  }
  
  # Combine segments with geometry
  combined_data <- do.call(rbind, segments_list)
  combined_data$geometry <- st_sfc(geometries_list, crs = st_crs(linestring))
  
  return(combined_data)
}

# Process each street individually to create uniform segment network
all_segments <- list()
segment_counter <- 0

for (i in 1:nrow(streets)) {
  current_road <- streets[i, ]
  road_geom <- st_geometry(current_road)[[1]]
  road_length <- current_road$length
  
  # Extract attributes excluding geometry and length
  road_attrs <- current_road %>% 
    st_drop_geometry() %>% 
    select(-length)
  
  # Segment each road
  road_segments <- segment_linestring(
    linestring = road_geom,
    road_length = road_length,
    max_length = 300,
    road_attributes = road_attrs
  )
  
  # Assign unique segment IDs across entire network
  road_segments$segment_id <- (segment_counter + 1):(segment_counter + nrow(road_segments))
  segment_counter <- segment_counter + nrow(road_segments)
  
  all_segments[[i]] <- road_segments
  
  # Progress tracking 
  if (i %% 1000 == 0) {
    cat("Processed", i, "of", nrow(streets), "roads\n")
  }
}

# Combine all segments
street_seg <- bind_rows(all_segments) %>% 
  st_as_sf() %>%
  select(-segment_length)

# ================= GEOGRAPHIC FILTERING =================

# Limit segments to Portland urban area using crash data extent
pdx_crashes <- crashes %>% filter(urb_area_short_nm == "PORTLAND UA")

street_seg <- st_crop(street_seg, 
                      c(xmin = min(pdx_crashes$longtd_dd), 
                        ymin = min(pdx_crashes$lat_dd),
                        xmax = max(pdx_crashes$longtd_dd), 
                        ymax = max(pdx_crashes$lat_dd)))

# Extract midpoint coordinates for each segment
segment_midpoints <- st_line_interpolate(st_geometry(street_seg), 0.5)
coords <- st_coordinates(segment_midpoints)

street_seg$lon <- coords[, "X"]
street_seg$lat <- coords[, "Y"]

# ================= WEATHER STATION ASSIGNMENT =================

# Assign nearest weather station to each segment for weather data linkage
add_weather_station <- function(segments, stations) {
  coords_segments <- cbind(segments$lon, segments$lat)
  coords_stations <- as.matrix(stations[, c("longitude", "latitude")])
  
  # Find nearest station using nearest neighbor search
  nearest_idx <- nn2(coords_stations, coords_segments, k = 1)$nn.idx[, 1]
  
  segments$location_id <- stations$location_id[nearest_idx]
  return(segments)
}

street_seg <- add_weather_station(street_seg, id_lookup)

# ================= EXPORT =================

# Convert to WKT format for storage
street_seg <- street_seg %>%
  mutate(geometry_wkt = st_as_text(geometry)) %>%  
  st_drop_geometry() %>%
  rename(geometry = geometry_wkt)

# Save final set of segments
write_parquet(street_seg, "../data/street_seg.parquet")
