log_file <- paste0("logs/log_", Sys.Date(), ".txt")

sink(log_file)

source("scripts/download_data.R")

source("scripts/clean_data.R")

sink()
