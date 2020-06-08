countLetters <- function(x, subset = FALSE){
  x <- unlist(strsplit(x, ''))
  alphList <- vector('list', 26)
  alphList[1:26] <- 0
  names(alphList) <- letters
  for(i in 1:length(x)){
    if(!(tolower(x[i]) %in% letters)){
      next
    } else {
      index <- match(tolower(x[i]), letters)
      alphList[[index]] <- alphList[[index]] + 1
    }
  }
  finalVec <- unlist(alphList)
  if(!subset) return(finalVec) else return(finalVec[finalVec != 0]) 
}

string <- paste0('hey thereaaaaaaaaaaaaaaaaaaafefffffffffffffffffffffeeeeeeeeeeeeeeeeeeeeeeeeeeeevvvvvvv',
                 "adgasdasdgasdgasdgasdgasdgasdasdgasdgasdgasdgafdbrgntyuimkjjnsfgbadskjanvarvar va'sda'",
                 "aienrf8922983u92ihenf;asjfoaksml ///\\  aosdjfha9wehf7h7289h 9898  noinsdfaaa")

countLetters(string)
sum(2+2)

h <- countLetters

