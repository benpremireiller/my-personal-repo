def focusString(string, keys, n = 50, prox = 25, sep = "..."):
    """
    This function looks for keys words in a string and isolates n/2 characters before and after the keys only
    if they are not in prox number of characters from a previously isolated word.
    """
    if type(keys) != list:
        raise BaseException("keys must be of type 'list'")
    
    focusStr = ""
    strLen = len(string)
    indexMemory = [strLen*2] #Initialize with large number so all() function works
    
    for key in keys:
        mentionCount = string.lower().count(key.lower())
        startIndex = 0
        
        for i in range(0,mentionCount):
            index = string.lower().find(key.lower(), startIndex)

            if all(abs(index - ind) >= prox for ind in indexMemory): 
                if (index - n) <= 0 and (index + n) > strLen:
                    focusStr = focusStr + string + sep
                elif (index - n) <= 0 and (index + n) <= strLen:
                    focusStr = focusStr + string[0:(index+n+1)] + sep
                elif (index - n) > 0 and (index + n) > strLen:
                    focusStr = focusStr + string[(index-n):(strLen+1)] + sep
                else:
                    focusStr = focusStr + string[(index-n):(index+n+1)] + sep

                indexMemory.append(index)
                startIndex += index+1

    return focusStr[0:len(focusStr)-len(sep)]

def focusString2(string, keys, n = 1, sep = "|~*~|"):
    #TODO: Maybe add checks for spaces after each period?
    """
    This function looks for keys words in a string and isolates n sentences (period to period) before and after
    each key word using sep as a seperator.
    """
    
    if type(keys) != list:
        raise BaseException("keys must be of type 'list'")

    focusStr = ""
    strLen = len(string)
    indexMemory = [[strLen*2, strLen*2]] #Initialize with large numbers so all() function works

    for key in keys:
        if key == '':
            continue
        mentionCount = string.lower().count(key.lower())
        startIndex = 0

        for i in range(0, mentionCount):
            index = string.lower().find(key.lower(), startIndex)

            if all(index < low or index > high and index != -1 for [low, high] in indexMemory):
                leftPeriodIndex = index
                rightPeriodIndex = index
                
                for j in range(0, n):
                    if leftPeriodIndex > 0:      
                        leftPeriodIndex = string.lower().rfind(".", 0, leftPeriodIndex-1)
                        
                    if rightPeriodIndex < strLen - 1:
                        if string[rightPeriodIndex + 1] == " ": #Check for spaces to the right of period
                            rightPeriodIndex += 1
                        rightPeriodIndex = string.lower().find(".", rightPeriodIndex)
                        
                    if rightPeriodIndex == -1: #If no period to the right
                        rightPeriodIndex = strLen - 1
                        
                    if j == 0: #No duplicates for main sentence
                        indexMemory.append([leftPeriodIndex, rightPeriodIndex])
                        if leftPeriodIndex == 0:
                            leftPeriodIndex = -1 #If key is element 0 of string (has to be a better way to do this)
                
                focusStr = focusStr + string[leftPeriodIndex+1:rightPeriodIndex+1] + sep   
                startIndex += index+1

    return focusStr[0:len(focusStr)-len(sep)]
                
                
