# ==============================================================================
# Traffic Crash Analysis Data Preprocessing Pipeline
# ==============================================================================
# Purpose: Prepare ML-ready dataset combining crash records with weather data
#          and generate negative samples for crash prediction modeling
# 
# Input Files:
#   - ../raw_data/weather.csv: Hourly weather observations
#   - ../data/id_lookup.csv: Weather station locations
#   - ../raw_data/crashes.csv: Traffic crash records
#   - ../data/street_seg.parquet: Street segments set
#
# Output:
#   - ../training/ml_input_data.parquet: ML-ready dataset with features
# ==============================================================================

# ================= LIBRARIES =================

library(data.table)
library(tidyverse)
library(lubridate)
library(suncalc)
library(sf)
library(RANN)
library(geosphere)
library(slider)
library(janitor)
library(fastDummies)
library(arrow)

# ================= WEATHER DATA PROCESSING =================

# Read and clean weather and station data
weather <- fread(
  "../raw_data/weather.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "NULL"),
  fill = TRUE,
  data.table = FALSE
) %>%
  clean_names()  

id_lookup <- fread(
  "../data/id_lookup.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "NULL"),
  fill = TRUE,
  data.table = FALSE
) %>%
  clean_names()  

# Parse datetime and set timezone
weather <- weather %>%
  mutate(datetime = ymd_hm(time)) %>%
  select(-time)

# Calculate 3-hour rolling precipitation by station
weather <- weather %>%
  arrange(location_id, datetime) %>%
  group_by(location_id) %>%
  mutate(
    rain_3hr = slide_index_dbl(
      .x = rain_inch,
      .i = datetime,
      .f = ~ sum(.x),
      .before = dhours(2),
      .complete = FALSE
    )
  ) %>%
  ungroup()

# Simplify column names and remove unused variables
weather <- weather %>%
  select(-weather_code_wmo_code, 
         -precipitation_inch,
         -wind_direction_10m,
         -sunshine_duration_s,
         -apparent_temperature_f, 
         -dew_point_2m_f, 
         -surface_pressure_h_pa,
         -is_day
         ) %>%
  rename(
    temp = temperature_2m_f,
    humidity = relative_humidity_2m_percent,
    rain = rain_inch,
    snow = snowfall_inch,
    snow_depth = snow_depth_ft,
    cloud_cover = cloud_cover_percent,
    wind_speed = wind_speed_10m_mp_h,
    wind_gusts = wind_gusts_10m_mp_h
  )

# ================= CRASH DATA PROCESSING =================

# Read and clean first crash data
crashes_1 <- fread(
  "../raw_data/crashes_1.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "NULL"),
  data.table = FALSE
) %>%
  clean_names()

# Filter to Portland and remove crashes with no location
crashes_1 <- crashes_1 %>%
  filter(
    urb_area_long_nm == "Portland UA", 
    unloct_flg == 0,
    crash_hr_no != 99
    ) 

# Extract datetime
crashes_1 <- crashes_1 %>%
  mutate(
    date_only = str_extract(crash_dt, "^\\d{4}/\\d{2}/\\d{2}"),
    hour_str = sprintf("%02d", crash_hr_no),
    datetime_str = paste(date_only, paste0(hour_str, ":00")),
    datetime = ymd_hm(datetime_str, tz = "America/Los_Angeles", quiet = TRUE)
  )

# Select relevant cols
crashes_1 <- crashes_1 %>%
  select(lat_dd, longtd_dd, datetime) %>%
  rename(
    lat = lat_dd,
    lon = longtd_dd
  )

# Read and clean second crash data
crashes_2 <- fread(
  "../raw_data/crashes_2.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "NULL"),
  data.table = FALSE
) %>%
  clean_names()

# Filter to Portland area and remove unusable records
crashes_2 <- crashes_2 %>%
  filter(
    urb_area_short_nm == "PORTLAND UA",
    unloct_flg == 0,
    crash_hr_no != 99
  ) 

# Extract datetimes
crashes_2 <- crashes_2 %>%
  mutate(
    hour_str = sprintf("%02d", crash_hr_no),
    datetime_str = paste(crash_dt, paste0(hour_str, ":00")),
    datetime = ymd_hm(datetime_str, tz = "America/Los_Angeles", quiet = TRUE) 
  )

# Select relevant cols
crashes_2 <- crashes_2 %>%
  select(lat_dd, longtd_dd, datetime) %>%
  rename(
    lat = lat_dd,
    lon = longtd_dd
  )

# Filter out overlap and incomplete year (2024) from crashes_2
crashes_2 <- crashes_2 %>%
  filter(datetime > "2022-12-31 23:59:59" 
         & datetime < "2023-12-31 23:59:59") 

# Join data
crashes <- bind_rows(crashes_1, crashes_2) %>%
  filter(!is.na(datetime))

# ================= CRASH STREET POINT ASSIGNMENT =================

# Load street segments
street_seg <- read_parquet("../data/street_seg.parquet") 

assign_nearest_street <- function(crashes, street_seg) {
  street_seg_sf <- street_seg %>%
    mutate(geometry = st_as_sfc(geometry, crs = 4326)) %>%
    st_as_sf()
  
  crash_sf <- st_as_sf(crashes, coords = c("lon", "lat"), crs = 4326)
  
  nearest_idx <- st_nearest_feature(crash_sf, street_seg_sf)
  
  crashes$type        <- street_seg$type[nearest_idx]
  crashes$location_id <- street_seg$location_id[nearest_idx]
  crashes$segment_id  <- street_seg$segment_id[nearest_idx]
  
  crashes
}

crashes <- assign_nearest_street(crashes, street_seg)

# Create positive samples dataset
positives <- crashes %>%
  mutate(crash_occurred = 1) %>%
  select(
    datetime,
    segment_id,
    type,
    crash_occurred,
    location_id
  ) 

# ================= SEGMENT STATISTICS =================

# Calculate segment-level crash frequency statistics 
all_segments <- street_seg %>%
  select(segment_id) %>%
  distinct()

segment_crash_stats <- positives %>%
  group_by(segment_id) %>%
  summarise(
    seg_count = n(),
    .groups = 'drop'
  )

# Ensure all segments have statistics (fill missing with 0)
segment_stats <- all_segments %>%
  left_join(segment_crash_stats, by = "segment_id") %>%
  mutate(
    seg_count = replace_na(seg_count, 0), # not used
    seg_freq = seg_count / nrow(positives), # not used
    seg_log_count = log1p(seg_count)
  ) %>%
  select(segment_id, seg_log_count) # Change this to include more stats

write_parquet(segment_stats, "../data/segment_stats.parquet")

# ================= STRATIFIED NEGATIVE SAMPLE GENERATION =================

# Set seed 
set.seed(123)

# Define time periods and target ratio
time_start <- as.POSIXct(min(crashes$datetime))
time_end   <- as.POSIXct(max(crashes$datetime))
target_ratio <- 5  # Target 5:1 ratio

# Add year column to positives for stratification
positives <- positives %>%
  mutate(
    datetime = as.POSIXct(datetime, tz = "America/Los_Angeles"),
    year = year(datetime)
  ) %>%
  filter(!is.na(year))  # Remove any rows with NA year

# Calculate positives and required negatives by year
positives_by_year <- positives %>%
  group_by(year) %>%
  summarise(
    n_positives = n(),
    .groups = 'drop'
  ) %>%
  mutate(
    n_negatives = n_positives * target_ratio
  )

# Generate negatives separately for each year to maintain consistent ratio
negatives_list <- list()

for (yr in positives_by_year$year) {
  
  year_info <- positives_by_year %>% filter(year == yr)
  n_needed <- year_info$n_negatives
  n_to_generate <- ceiling(n_needed * 1.1)
  
  # Define year-specific time range
  year_start <- as.POSIXct(sprintf("%d-01-01 00:00:00", yr), tz = "America/Los_Angeles")
  year_end <- as.POSIXct(sprintf("%d-12-31 23:00:00", yr), tz = "America/Los_Angeles")
  year_end <- min(year_end, time_end)
  
  # Generate all available hours for this year
  year_hours <- seq(year_start, year_end, by = "hour")
  
  # Sample indices
  sampled_indices <- sample(1:length(year_hours), n_to_generate, replace = TRUE)
  sampled_datetimes <- year_hours[sampled_indices]
  
  # Sample random segment-time combinations for this year
  year_negatives <- street_seg %>%
    sample_n(n_to_generate, replace = TRUE) %>%
    mutate(
      datetime = sampled_datetimes,
      year = yr
    )
  
  # Remove any overlaps with actual crashes in this year
  year_positives <- positives %>% 
    filter(year == yr) %>%
    select(segment_id, datetime)
  
  year_negatives <- year_negatives %>%
    anti_join(year_positives, by = c("segment_id", "datetime")) %>%
    distinct(segment_id, datetime, .keep_all = TRUE)
  
  if (nrow(year_negatives) >= as.integer(n_needed)) {
    year_negatives <- year_negatives %>% sample_n(as.integer(n_needed))
  }
  
  negatives_list[[as.character(yr)]] <- year_negatives
}

# Combine negatives
negatives <- bind_rows(negatives_list) %>%
  mutate(crash_occurred = 0) %>%
  select(datetime, segment_id, type, crash_occurred, location_id)
  
# Drop year
positives <- positives %>% select(-year)

# ================= FINAL DATA SET CREATION =================

# Combine positives and negatives
ml_input_data <- bind_rows(positives, negatives)

# Expand date features
ml_input_data <- ml_input_data %>%
  mutate(
    month = month(datetime),
    day = wday(datetime, week_start = 7),
    hour = hour(datetime)
  )

# Add weather features
ml_input_data <- ml_input_data %>%
  left_join(weather, by = c("location_id", "datetime")) %>%
  select(-location_id) %>%
  filter(!is.na(temp))

# Convert categoricals to factors and one hot encode
ml_input_data <- ml_input_data %>%
  mutate(across(c("month", "day", "hour", "type"), as.factor))

ml_input_data <- ml_input_data %>%
  dummy_cols(remove_selected_columns = TRUE,  remove_first_dummy = FALSE) %>%
  arrange(datetime)

# Save final model input
write_parquet(ml_input_data, "../training/ml_input_data.parquet")
