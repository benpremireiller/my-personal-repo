rm(list=ls())
library(rvest)
library(dplyr)
library(stringr)
library(xlsx)
library(FAO)

#####################********INPUTS*********#####################

#URL of table of contents or page where you want to start scraping
link <- "http://budget.ontario.ca/2019/foreword.html"

#####################**********************#####################

#Parsing----------------
html <- read_html(link)

#Isolate domain name/budget year
lastSlash <- unlist(str_locate_all(link, "/"))
linkInd <- lastSlash[length(lastSlash)]
linkRoot <- str_sub(link, 1, linkInd)

#Finding 'next' button url
href <- html %>% 
  html_nodes(xpath = "//*[@direction='next']/a") %>%
  html_attr("href")

tables <- vector('list', 100) #Ensure length is over total # of tables, currently arbitrarily set to 100
loop <- 0

while(length(href != 0)){ #While there is a next button
  
  tbl <- html %>% 
    html_nodes(xpath = "//table/@id") %>% 
    html_text()
  
  tblNs <- NULL
  for(i in 1:length(tbl)){
    tblNs[i] <- html %>%
      html_nodes(xpath = paste0("//table[@id='",tbl[i],"']/@summary")) %>% 
      html_text()
  }
  
  if(length(tbl) != 0){ #Only loop if tables exist
    
    loop <- loop + 1
    
    #Find total rows of all tables on current page
    rcount <- sapply(seq_along(tbl), function(x){
      length(html %>% 
               html_nodes(css = paste("table#",tbl[x]," > tbody > tr", sep=""))) 
    })
    #Find column count of all tables on current page
    colcount <- sapply(seq_along(tbl), function(x){
      length(html %>% 
               html_nodes(xpath = paste0("//table[@id='",tbl[x],"']/thead/tr/child::*")))
    })
    
    if(loop == 1){
      tblIndex <- 0  #Initialize index. Used to append tables to final list of all tables.
      tblNames <- NULL #Initalize table name vector
    }
    
    #Loop through all tables on current page
    for(x in 1:length(tbl)){
      tables[[x + tblIndex]] <- matrix(nrow = rcount[x], ncol = colcount[x])
      dftitles <- NULL 
      for(j in 1:colcount[x]){
        dftitles[j] <- tryCatch(dftitles[j] <- html %>% #Table headers
          html_nodes(xpath = paste0("//table[@id='",tbl[x],"']/thead/tr/child::*[",j,"]")) %>%
          html_text(), error = function(e) print("Error"))
        if(j == 1){ #First column (row names)
          tables[[x + tblIndex]][,j] <-  tryCatch(tables[[x + tblIndex]][,j] <- html %>% 
                      html_nodes(css = paste0("table#",tbl[x]," > tbody > tr > th")) %>% 
                      html_text() %>% 
                      as.character() %>% 
                      str_replace_all("\\n|\\r","") %>% 
                      str_squish(), error = function(e) print("Error"))
        } else {  #Data values
          tables[[x + tblIndex]][,j] <- tryCatch(tables[[x + tblIndex]][,j] <- html %>% #Actual values
             html_nodes(css = paste("table#",tbl[x]," > tbody > tr > td:nth-child(",j,")", sep="")) %>% 
             html_text() %>% 
             as.character() %>% 
             str_trim(), error = function(e) print("Error"))
        }
      }
      tables[[x + tblIndex]] <- data.frame(tables[[x + tblIndex]]) #Convert matrix to dataframe
      dftitles <- gsub("[[:space:]]", "", dftitles) #Remove all spaces in titles
      names(tables[[x + tblIndex]]) <- dftitles #Assign variable names
    }
  tblIndex <- tblIndex + length(colcount) #Could use row count
  tblNames <- c(tblNames, tblNs)
  }
  
  #Find next button
  href <- html %>% 
    html_nodes(xpath = "//*[@direction='next']/a") %>%
    html_attr("href")
  
  #If next button exists, read new webpage
  if(length(href) != 0){
    link <- paste0(linkRoot, href[1])
    html <- read_html(link)
  }
}


#Remove NULLs in list
tables <- tables[1:tblIndex]

#Assign table names
allTblNames <- NULL
for(i in 1:length(tblNames)) allTblNames[i] <- condense_str(rm_invalid(proper(tblNames[i])), 31, T, T, rm.words = c("Table","Summary Of", "Ontario's"))
#Not removing "Ontario's": MAKES NO SENSE because it works if you just run it alone???
names(tables) <- allTblNames

#write to xlsx
setwd("path")
filename <- "alltables.xlsx" #set this to desired file name

if(file.exists(filename)) file.remove(filename)

for(i in 1:length(tables)){
  write.xlsx(as.data.frame(tables[[i]]), file = filename, sheetName = allTblNames[i], append = T, row.names = F)
}
