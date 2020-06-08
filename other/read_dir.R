read_dir <- function (path = getwd(), file.type = c("csv"), read.as = c("csv")) 
{
  map <- matrix(c("csv", "tsv", "xlsx",
                  ",",   "\t",  "xlsx"), ncol = 2)
  
  if(length(file.type) != length(read.as)) stop("file.type and read.as must be of same length")
  if(!(all(read.as %in% map[, 1]))) stop("A read.as type specified is not supported")
  
  fileNames <- list.files(path)
  allTbls <- vector('list', length(fileNames))
  
  read_type <- function(li, file.type, read.as, ind) {
    if (length(file.type) == 0) 
      return(li[1:ind-1])
    
    typeFiles <- fileNames[grepl(paste0("(.)+\\.", file.type[1]), fileNames)]
    incr <- length(typeFiles)
    
    tblList <- lapply(typeFiles, function(f) {
      if (read.as[1] == "xlsx") {
        readxl::read_xlsx(paste0(path, "/", f), 1)
      }
      else {
        utils::read.table(paste0(path, "/", f), header = TRUE, sep = map[map[, 1] == read.as[1]][2])
      }
    })
    
    li[ind:(ind+incr-1)] <- tblList
    names(li)[ind:(ind+incr-1)] <- gsub("\\.(.)+", "", typeFiles)
    
    read_type(li, file.type[-1], read.as[-1], ind+incr)
  }
  
  return(read_type(allTbls, file.type, read.as, 1))
}

setwd("./Test")
a <- read_dir(file.type = c("csv", "txt", "xlsx"), read.as = c("csv", "tsv", "xlsx"))
