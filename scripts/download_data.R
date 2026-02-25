library(httr)
library(data.table)
library(yaml)

config<-read_yaml("covid_config.yml")
download_dataset <- function(name, info) {
  
  date_stamp <- Sys.Date()
  
  destfile <- paste0(
    config$output$raw_dir,
    info$file_prefix,
    "_",
    date_stamp,
    ".csv"
  )
  
  GET(info$url, write_disk(destfile, overwrite = TRUE))
  
  return(destfile)
  
}

files <- mapply(
  download_dataset,
  names(config$sources),
  config$sources
)
