library(psymonitor)
library(ggplot2)
library(dplyr)

bubble_raw <- read.csv("bubble_raw.csv", stringsAsFactors = FALSE)
bubble_raw$date <- as.Date(bubble_raw$date)

bubble_raw <- bubble_raw[
  bubble_raw$date >= as.Date("1990-01-31") &
  bubble_raw$date <= as.Date("2025-12-31"),
]

y <- bubble_raw$log_p

obs      <- length(y)
r0       <- 0.01 + 1.8 / sqrt(obs)
swindow0 <- floor(r0 * obs)
dim      <- obs - swindow0 + 1

IC     <- 2      # BIC
adflag <- 6      # max lag
yr     <- 2
Tb     <- 12 * yr + swindow0 - 1
nboot  <- 99     # start with 49 if slow

bsadf <- PSY(y, swindow0, IC, adflag)

quantilesBsadf <- cvPSYwmboot(
  y,
  swindow0,
  IC,
  adflag,
  Tb,
  nboot,
  nCores = 2
)

monitorDates <- bubble_raw$date[swindow0:obs]

quantile95 <- quantilesBsadf %*% matrix(1, nrow = 1, ncol = dim)
ind95      <- (bsadf > t(quantile95[2, ])) * 1

periods <- locate(ind95, monitorDates)
bubbleDates <- disp(periods, obs)

print(bubbleDates)

write.csv(
  bubbleDates,
  file = "label_dates.csv",
  row.names = FALSE
)