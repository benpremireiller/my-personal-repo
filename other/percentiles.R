pctl <- function(vec, n){
  n <- seq(0, 1, 1/n)
  pctl <- quantile(vec, n)[-1]
  names(pctl) <- 1:length(pctl)
  return(pctl)
}

# pctl.match <- function(vec, n){
#   matchVec <- NULL
#   pctl <- percentile(vec, n)
#   for(i in 1:length(vec)){
#     check <- vec[i] <= pctl
#     matchVec[i] <- names(pctl[check][1])
#   }
#   return(matchVec)
# }

pctl.weighted <- function(vec, weights = rep(1, length(vec)), n = 100, fun = 'mean'){
  if(length(vec) != length(weights)) stop("Vector and weights must be of same length.")
  if(fun != 'mean' && fun != 'median') stop("Invalid 'fun' argument: this function currently only supports either mean or median.")
  
  df <- data.frame(val = vec, wgt = weights)
  df <- df[order(df$val),]
  df$cum <- cumsum(df$wgt)
  totalWght <- sum(df$wgt)
  intervals <- totalWght / n
  pctl <- seq(0, totalWght, by = intervals)
  
  output <- NULL
  for(j in 1:(length(pctl)-1)){
    diff1 <- df$cum - pctl[j]
    diff2 <- df$cum - pctl[j+1]
    pos1 <- which(diff1 >= 0)[1]
    pos2 <- which(diff2 >= 0)[1]
    
    if(fun == 'mean'){
      nPerObs <- NULL
      wghts <- NULL
      for(i in pos1:pos2){
        if(i == pos1){
          nPerObs[i-pos1+1] <- min(diff1[i], intervals)
        } else if(i == pos2){
          nPerObs[i-pos1+1] <- intervals - sum(nPerObs)
        } else {
          nPerObs[i-pos1+1] <- df$wgt[i]
        }
      }
      wghts <- nPerObs/intervals
      output[j] <- sum(wghts*df$val[pos1:pos2])
      
    } else if(fun == 'median'){ #TODO: Add average of 'middle' values
      medInter <- pctl[j]+(intervals/2)
      for(i in pos1:pos2){ #Something like: if pos2-pos1 %% 2 == 0 
        if(medInter <= df$cum[i]){
          output[j] <- df$val[i]
          break
        } else if(medInter > df$cum[i] && medInter <= df$cum[i+1]){
          output[j] <- df$val[i+1]
          break
        }
      }
    }
  }
  names(output) <- 1:n
  return(output)
}

pctl.weighted(1:101, rep(1,101), 1, 'mean') == weighted.mean(1:101, rep(1,101))
pctl.weighted(5:10110, seq(5, 10110, 1), 100, 'mean')
mean(1:100)
median(1:100)
