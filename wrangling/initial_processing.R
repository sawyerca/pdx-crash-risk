# ==============================================================================
# Traffic Crash Analysis Data Preprocessing Pipeline
# ==============================================================================
# Purpose: Prepare ML-ready dataset combining crash records with weather data
#          and generate negative samples for crash prediction modeling
# 
# Input Files:
#   - weather.csv: Hourly weather observations
#   - ../data/id_lookup.csv: Weather station locations
#   - crashes.csv: Traffic crash records
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
  "weather.csv",
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
  mutate(datetime = with_tz(datetime, tzone = "America/Los_Angeles")) %>%
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
      .before = dhours(3),
      .complete = FALSE
    )
  ) %>%
  ungroup()

# Add daylight indicator using regional center coordinates
mean_lat = mean(id_lookup$latitude)
mean_lon = mean(id_lookup$longitude)

weather <- weather %>%
  mutate(
    date_local = as.Date(datetime, tz = "America/Los_Angeles"),
    sun_times = getSunlightTimes(
      date = date_local,
      lat  = mean_lat,
      lon  = mean_lon,
      tz   = "America/Los_Angeles"
    ),
    sunrise = sun_times$sunrise,
    sunset  = sun_times$sunset,
    is_day  = as.numeric(datetime >= sunrise & datetime <= sunset)
  ) %>%
  select(-sun_times, -date_local, -sunrise, -sunset)

# Simplify column names and remove unused variables
weather <- weather %>%
  select(-weather_code_wmo_code, 
         -wind_direction_10m,
         -apparent_temperature_f, 
         -dew_point_2m_f, 
         -surface_pressure_h_pa) %>%
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

# Read and clean crash data
crashes <- fread(
  "crashes.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "NULL"),
  data.table = FALSE
) %>%
  clean_names()

# Filter to Portland area and remove unusable records
crashes <- crashes %>%
  filter(
    urb_area_short_nm == "PORTLAND UA",
    unloct_flg == 0,
    crash_hr_no != 99
  ) %>%
  mutate(
    hour_str = sprintf("%02d", crash_hr_no),
    datetime_str = paste(crash_dt, paste0(hour_str, ":00")),
    datetime = ymd_hm(datetime_str) %>% 
      force_tz(tzone = "America/Los_Angeles") 
  ) %>%
  select(-hour_str, -datetime_str)

# ================= CRASH STREET POINT ASSIGNMENT =================

# Load street segments
street_seg <- read_parquet("../data/street_seg.parquet") 

# Assign each crash to nearest street segment using nearest neighbor
assign_nearest_street <- function(crashes, street_seg) {
  crash_mat  <- as.matrix(crashes[, c("longtd_dd", "lat_dd")])
  street_mat <- as.matrix(street_seg[, c("lon", "lat")])
  
  nearest <- nn2(street_mat, crash_mat, k = 1)
  idx <- nearest$nn.idx[, 1]
  
  crashes$type <- street_seg$type[idx]
  crashes$location_id <- street_seg$location_id[idx]
  crashes$segment_id <- street_seg$segment_id[idx]
  
  crashes
}

crashes <- assign_nearest_street(crashes, street_seg)

# Drop lat/lon after weather station assignment is complete
street_seg <- street_seg %>%
  select(-lat, -lon)

# Rewrite street_seg without lat/lon
write_parquet(street_seg, "../data/street_seg.parquet")

# Create positive samples dataset
positives <- crashes %>%
  mutate(crash_occurred = 1) %>%
  select(
    datetime,
    segment_id,
    crash_mo_no,
    crash_wk_day_cd,
    crash_hr_no,
    type,
    crash_occurred,
    crash_svrty_short_desc,
    location_id
  )  %>%
  rename(
    month = crash_mo_no,
    day = crash_wk_day_cd,
    hour = crash_hr_no,
    svrty = crash_svrty_short_desc
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

# ================= NEGATIVE SAMPLE GENERATION =================

# Generate negative samples at 10:1 ratio to handle class imbalance
set.seed(123)
sample_ratio = 10 * nrow(positives)
num_gen = sample_ratio * 1.001 # Slight oversample to account for overlap removal

# Time range
time_start <- as.POSIXct("2019-01-01 00:00:00", tz = "America/Los_Angeles")
time_end   <- as.POSIXct("2024-12-31 15:00:00", tz = "America/Los_Angeles")

# Sample random segment-time combinations, remove overlaps with actual crashes
negatives <- street_seg %>%
  sample_n(num_gen, replace = TRUE) %>%
  mutate(datetime = sample(
    seq(time_start, time_end, by = "hour"), 
    num_gen, replace = TRUE
  )
  ) %>% 
  anti_join(positives, by = c("segment_id", "datetime")) %>%
  sample_n(sample_ratio)

# Finalize features and clean up
negatives <- negatives %>%
  mutate(
    month = month(datetime),
    day = lubridate::wday(datetime, week_start = 7),
    hour = hour(datetime),
    crash_occurred = 0,
    svrty = NA
  ) %>%
  select(-full_name, -geometry)

# ================= FINAL DATA SET CREATION =================

# Combine positives and negatives, add weather features
ml_input_data <- bind_rows(positives, negatives)

ml_input_data <- ml_input_data %>%
  left_join(weather, by = c("location_id", "datetime")) %>%
  select(-location_id) %>%
  filter(!is.na(temp))

# Convert categoricals to factors and create dummy variables
ml_input_data <- ml_input_data %>%
  mutate(across(c("month", "day", "hour", "type", "svrty"), as.factor))

ml_input_data <- ml_input_data %>%
  dummy_cols(remove_selected_columns = TRUE,  remove_first_dummy = FALSE) %>%
  arrange(datetime)

# Save final model input
write_parquet(ml_input_data, "../training/ml_input_data.parquet")
