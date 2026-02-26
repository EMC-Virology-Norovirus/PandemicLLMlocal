library(plotly)
library(data.table)
library(htmlwidgets)

find_latest_gauge <- function() {
  candidates <- c()
  top_level <- file.path("results", "gauge_output.csv")
  if (file.exists(top_level)) {
    candidates <- c(candidates, top_level)
  }
  run_files <- list.files(
    path = file.path("results", "runs"),
    pattern = "gauge_output\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  candidates <- c(candidates, run_files)
  if (length(candidates) == 0) {
    stop("No gauge_output.csv files found under results/ or results/runs/.")
  }
  info <- file.info(candidates)
  candidates[which.max(info$mtime)]
}

gauge_path <- find_latest_gauge()
cat(sprintf("Using gauge data: %s\n", gauge_path))
dat_f <- fread(gauge_path)
fig <- plot_ly()

risk_color <- function(score){
  
  if(score <= 20) return("darkgreen")
  else if(score <= 40) return("green")
  else if(score <= 60) return("grey")
  else if(score <= 80) return("orange")
  else return("red")
  
}

for(i in 1:nrow(dat_f)){
  cat_col <- risk_color(dat_f[horizon==i, risk_score_1_100])
  
  fig <- fig %>% add_trace(
    
    type = "indicator",
    mode = "gauge+number",
    
    value = dat_f[horizon==i, risk_score_1_100],
    
    domain = list(row = 0, column = i-1),
    
    title = list(
      text = paste0(
        
        "<span style='font-size:20px'><b>Horizon ", i, "</b></span><br>",
        
        "<span style='font-size:16px;color:gray'>",
        dat_f[horizon==i, date],
        "</span><br>",
        
        "<span style='color:", cat_col, "'><b>",
        toupper(dat_f[horizon==i, category]),
        "</b></span>"
        
      )
    ),
    
    gauge = list(
      
      axis = list(range = c(0, 100)),
      
      bar = list(color = "black"),
      
      steps = list(
        list(range = c(0, 20), color = "darkgreen"),
        list(range = c(20, 40), color = "green"),
        list(range = c(40, 60), color = "yellow"),
        list(range = c(60, 80), color = "orange"),
        list(range = c(80, 100), color = "red")
      ),
      
      threshold = list(
        line = list(color = "black", width = 6),
        value = dat_f[horizon==i, risk_score_1_100]
      )
    )
  )
}

fig <- fig %>% layout(
  
  grid = list(
    rows = 1,
    columns = nrow(dat_f),
    pattern = "independent"
  ),
  
  margin = list(t = 120)
)

out_html <- file.path("results", "gauge_plot.html")
saveWidget(fig, out_html, selfcontained = TRUE)
cat(sprintf("Saved %s\n", out_html))
