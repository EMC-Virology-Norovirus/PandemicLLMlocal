library(tidyverse)
library(janitor)
library(arrow)
library(data.table)
library(tsibble)
library(zoo)

raw_files <- list.files(
  "data/raw/",
  full.names = TRUE
)

pick_latest_file <- function(pattern, label) {
  matches <- raw_files[grepl(pattern, basename(raw_files), ignore.case = TRUE)]
  if (length(matches) == 0) {
    stop(paste0("No raw file found for ", label, " (pattern: ", pattern, ")."))
  }
  matches[which.max(file.info(matches)$mtime)]
}

ww_file <- pick_latest_file("ww_", "wastewater")
variant_file <- pick_latest_file("variant", "variant")
case_file <- pick_latest_file("case", "case")
pos_file <- pick_latest_file("positiv", "positivity")


ww <- fread(ww_file)
variant<- fread(variant_file)
case <- fread(case_file)
pos <- fread(pos_file)


date_top_melt<-function(data, value_var='count'){
  data_m<-melt(data, id='V1')
  data_m[,EpiWeek:=as.numeric(gsub("wk ","",V1))]
  data_m[variable=='2021/2022' & EpiWeek>=40 & EpiWeek<=52, Year:=2021]
  data_m[variable=='2021/2022' & EpiWeek<40, Year:=2022]
  data_m[variable=='2022/2023' & EpiWeek>=40 & EpiWeek<=52, Year:=2022]
  data_m[variable=='2022/2023' & EpiWeek<40, Year:=2023]
  data_m[variable=='2023/2024' & EpiWeek>=40 & EpiWeek<=52, Year:=2023]
  data_m[variable=='2023/2024' & EpiWeek<40, Year:=2024]
  data_m[variable=='2024/2025' & EpiWeek>=40 & EpiWeek<=52, Year:=2024]
  data_m[variable=='2024/2025' & EpiWeek<40, Year:=2025]
  data_m[variable=='2025/2026' & EpiWeek>=40 & EpiWeek<=52, Year:=2025]
  data_m[variable=='2025/2026' & EpiWeek<40, Year:=2026]
  data_m[,yearweek:=make_yearweek(year=Year, week=EpiWeek, week_start = 7)]
  data_m[,date:=as.Date(yearweek)]
  data_m[,`:=`(yearweek=NULL, V1=NULL, variable=NULL)]
  data_m[,download_date:=Sys.Date()]
  data_m2<-data_m[!is.na(value)]
  setnames(data_m2, 'value', value_var)
  return(data_m2)
}

variant_top_melt<-function(data){
  data2<-copy(data)
  data2[,week:=str_sub(V1,7.8)]
  data2[,year:=str_sub(V1,1,4)]
  data2[,yearweek:=make_yearweek(year=as.numeric(year), week=as.numeric(week), week_start = 7)]
  data2[,date:=as.Date(yearweek)]
  data2[,`:=`(V1=NULL, year=NULL, week=NULL, yearweek=NULL)]
  colnames(data2) <- paste("Lineage", colnames(data2), sep = "_")
  setnames(data2,"Lineage_date",'date')
  return(data2)
}
  

ww_c<-date_top_melt(ww, value_var='concentration')
pos_c<-date_top_melt(pos, value_var='pos_rate')
case_c<-date_top_melt(case, value_var='cases')
variant_c<-variant_top_melt(variant)

##add segment on Google Search Trends if I am chosen to be alpha tester

#write cleaned individual files to processed data

write_parquet(ww_c, "data/processed/COVID_ww.parquet")
write_parquet(pos_c, "data/processed/COVID_positivity.parquet")
write_parquet(case_c, "data/processed/COVID_cases.parquet")
write_parquet(ww_c, "data/processed/COVID_ww.parquet")



##now clean and combine all data and create smoothed and lagged variables
# -----------------------------
cases_ts <- case_c %>%
  group_by(date = floor_date(date, unit = "week")) %>%
  summarise(cases = sum(cases, na.rm = TRUE)) %>%
  arrange(date)

ww_weekly <- ww_c %>%
  group_by(date = floor_date(date, "week")) %>%
  summarise(ww_mean = mean(concentration, na.rm = TRUE)) %>%
  arrange(date) %>%
  mutate(ww_smooth = rollapply(ww_mean, width = 3, FUN = mean, fill = NA, align = "right"))%>%
  filter(!is.na(ww_mean))

# # C) Social media features: weekly counts and leading indicator (lag)
# social_weekly <- social %>%
#   group_by(date = floor_date(date, "week")) %>%
#   summarise(social_index = mean(average_search, na.rm = TRUE)) %>%
#   arrange(date) %>%
#   mutate(social_lag1 = lag(social_index, 1))

lineage_freq <- variant_c%>%
  mutate(week = floor_date(date, "week"))

pct_ts <- pos_c %>%
  group_by(date = floor_date(date, "week")) %>%
  summarise(pct_mean = mean(pos_rate, na.rm = TRUE)) %>%
  arrange(date) %>%
  mutate(pos_lag1 = lag(pct_mean, 1))


# Merge everything into modeling frame
df <- cases_ts %>%
  left_join(ww_weekly, by = "date") %>%
  #left_join(social_weekly, by = "date") %>%
  left_join(lineage_freq, by = c("date" = "week")) %>%
  left_join(pct_ts, by = "date") %>%
  arrange(date)

df <- df %>%
  mutate(cases_lag1 = lag(cases, 1),
         cases_lag2 = lag(cases, 2),
         week_num = row_number())


add_lineage_moving_averages <- function(
    df,
    lineage_prefix = "Lineage_",
    window = 4,
    align = "right",
    suffix = NULL
) {
  
  if (is.null(suffix)) {
    suffix <- paste0("_ma", window)
  }
  
  lineage_cols <- names(df)[startsWith(names(df), lineage_prefix)]
  
  df %>%
    arrange(date) %>%
    mutate(
      # 4) trailing moving average
      across(
        all_of(lineage_cols),
        ~ zoo::rollapply(
          .x,
          width = window,
          FUN = mean,
          align = align,
          fill = 0,
          na.rm = TRUE
        ),
        .names = paste0("{.col}", suffix)
      )
    )
}



df_ma <- add_lineage_moving_averages(
  df,
  lineage_prefix = "Lineage_",
  window = 4
)

lineage_prefix = "Lineage_"
lineage_cols <- names(df)[startsWith(names(df), lineage_prefix)]
lineage_ma_cols<-paste0(lineage_cols, "_ma4")

df_ma <- df_ma %>%  mutate(
  across(
    all_of(lineage_ma_cols),
    ~ ifelse(is.na(.x), 0, .x)
  )
)

#now drop the original columns and rename the ma ones:
df_ma2 <- df_ma %>% select(-lineage_cols)
df_ma2 <- df_ma2 %>%
  rename_with(
    ~ sub("_ma4$", "", .x),
    ends_with("_ma4")
  )


# remove early incomplete rows and crop back to have hold outs for forecasting:
df_model <- df_ma2 %>% filter(!is.na(cases))
df_model <- df_model %>% select(-date.y)
df_model_hist <- df_model %>% filter(date<=max(df_model$date)-21)
df_model_test <- df_model %>% filter(date>max(df_model$date)-21)

fwrite(df_model_hist,"data/processed/surveillance_COVID19_weekly.csv")
fwrite(df_model_test[,c('date','cases')],"data/processed/validate_csv.csv")
