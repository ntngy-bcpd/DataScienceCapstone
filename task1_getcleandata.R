## Data Science Capstone Project
## Coursera-Swiftkey Prediction Model

#################
## Load libraries
#################

library(stringi)
library(caret)
library(tm)
library(qdap)
library(wordcloud)
library(ngram)

## Set the default number of threads to use
## To avoid the hang problem with NGramTokenizer
## from https://stackoverflow.com/questions/19024873/why-does-r-hang-when-using-ngramtokenizer
# options(mc.cores=1)


#################################
## Set seed and working directory
#################################

set.seed(20170801)
setwd("C:/Courses/Data10 Data Science Capstone Project/Capstone")


###########
## Get data
###########

## readTextFile: Reads a text file to a vector whose elements are lines 
## in the file.
##
## fileName = input text fileName in the working directory
## warn = FALSE, skip warnings of missing EOF
## skipNul = TRUE, skip warnings of embedded nul
## textVector = output a vector of text
##
readTextFile <- function(fileName, warn=FALSE, skipNul=TRUE) {
    textFile <- file(fileName, open="r")
    textVector <- readLines(textFile, warn=warn, skipNul=skipNul)
    close(textFile)
    return(textVector)
}

## Download the Coursera-Swiftkey.zip file from the Coursera website
## and unzip the folder Coursera-Swiftkey to the working directory

## Read blogs 
blogsFileName <- ".\\Coursera-Swiftkey\\final\\en_US\\en_US.blogs.txt"
blogs <- readTextFile(blogsFileName)

## Read news
newsFileName <- ".\\Coursera-Swiftkey\\final\\en_US\\en_US.news.txt"
news <- readTextFile(newsFileName)

## Read twitters
twitterFileName <- ".\\Coursera-Swiftkey\\final\\en_US\\en_US.twitter.txt"
twitter <- readTextFile(twitterFileName)

## Count lines in sources
blogsLines <- length(blogs)
newsLines <- length(news)
twitterLines <- length(twitter)
totalLines <- blogsLines + newsLines + twitterLines

## Count words in sources
## Require *** library stringi
blogsWords <- sum(stri_count_words(blogs))
newsWords <- sum(stri_count_words(news))
twitterWords <- sum(stri_count_words(twitter))
totalWords <- blogsWords + newsWords + twitterWords

## Summarize sources
sourceLines <- c(blogsLines, newsLines, twitterLines, totalLines)
sourceWords <- c(blogsWords, newsWords, twitterWords, totalWords)
sourceWordsPerLine <- c(blogsWords/blogsLines,
                        newsWords/newsLines,
                        twitterWords/twitterLines,
                        totalWords/totalLines
                        )
sourceSummary <- data.frame(sourceLines, sourceWords, sourceWordsPerLine)
rownames(sourceSummary) <- c("Blogs", "News", "Twitter", "Total")
colnames(sourceSummary) <- c("Number_of_Lines", "Number_of_Words", "Words_per_Line")
sourceSummary


########################
## Sample and Split Data
########################

## Sampling factor for sample size = samplingFactor * total number of lines
samplingFactor = 0.1

## Training ratio
trainPercent <- 0.6
validatePercent <- 0.2
testPercent <- 1 - trainPercent - validatePercent
testSplit <- testPercent
trainSplit <- trainPercent/(trainPercent + validatePercent)

## Sample and combine the sources with a sampling factor
blogsSample <- sample(blogs, size=length(blogs)*samplingFactor, replace=FALSE)
newsSample <- sample(news, size=length(news)*samplingFactor, replace=FALSE)
twitterSample <- sample(twitter, size=length(twitter)*samplingFactor, replace=FALSE)

## Combine samples and permutate
rawData <- c(blogsSample, newsSample, twitterSample)
rawData <- sample(rawData, size=length(rawData), replace=FALSE)

## Split the raw data into test and non-test data sets
## *** Require library caret
inTest <- createDataPartition(seq_len(NROW(rawData)), p=testSplit, list=FALSE)
rawTestData <- rawData[inTest]
rawNonTestData <- rawData[-inTest]

## Split the non-test data into training and validation data sets
inTrain <- createDataPartition(seq_len(NROW(rawNonTestData)), p=trainSplit, list=FALSE)
rawTrainData <- rawNonTestData[inTrain]
rawValidateData <- rawNonTestData[-inTrain]

## Remove auxiliary data
rm(inTest, inTrain)

## Count lines in data sets
trainLines <- length(rawTrainData)
validateLines <- length(rawValidateData)
testLines <- length(rawTestData)
rawLines <- trainLines + validateLines + testLines

## Count words in data sets
## Require library stringi
trainWords <- sum(stri_count_words(rawTrainData))
validateWords <- sum(stri_count_words(rawValidateData))
testWords <- sum(stri_count_words(rawTestData))
rawWords <- trainWords + validateWords + testWords

## Summarize data sets
datasetLines <- c(trainLines, validateLines, testLines, rawLines)
datasetWords <- c(trainWords, validateWords, testWords, rawWords)
datasetWordsPerLine <- c(trainWords/trainLines,
                         validateWords/validateLines,
                         testWords/testLines,
                         rawWords/rawLines
                         )
datasetSummary <- data.frame(datasetLines, datasetWords, datasetWordsPerLine)
rownames(datasetSummary) <- c("Training", "Validation", "Testing", "Total")
colnames(datasetSummary) <- c("Number_of_Lines", "Number_of_Words", "Words_per_Line")
datasetSummary


#############
## Clean data
#############

## preprocessData: clean a corpus by converting its text to plain text document
## and lower case, replacing contractions with their full forms, and
## removing profanities, numbers and punctuation. Removing English stopwords
## is optional.
##
## rawCorpus = input text to be preprocessed.
## isStopWordsRemoved = FALSE, set to TRUE to remove stopwords
## cleanCorpus = output preprocessed corpus
##
preprocessData <- function(rawCorpus, isStopWordsRemoved=FALSE) {
    
    if (!require(tm)) {
        stop("Library tm is missing.")
    }
    ## Convert to plain text
    cleanCorpus <- tm_map(rawCorpus, PlainTextDocument)
    
    ## Replace contractions with their full form using qdap dictionary
    ## from https://trinkerrstuff.wordpress.com/my-r-packages/qdap/
    if (!require(qdap)) {
        stop("Library qdap is missing.")
    }
    cleanCorpus <- tm_map(cleanCorpus, content_transformer(replace_contraction))
    
    ## Convert to lower case
    cleanCorpus <- tm_map(cleanCorpus, content_transformer(tolower))
    
    ## Read Google naughty word list
    ## from https://gist.github.com/ryanlewis/a37739d710ccdb4b406d
    googleBwlFileName <- ".\\google_twunter_lol"
    google_twunter_lol <- readTextFile(googleBwlFileName)

    ## Remove profanities defined in Google list
    cleanCorpus <- tm_map(cleanCorpus, removeWords, google_twunter_lol)
    
    ## Remove stopwords
    if (isStopWordsRemoved) {
        cleanCorpus <- tm_map(cleanCorpus, removeWords, stopwords("english"))
    }
    
    ## Remove numbers, puntuation and strip white space
    
    ## Replace non alpha-numeric characters with spaces
    ## toSpace from onepager.togaware.com/TextMiningO.pdf
    toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
    cleanCorpus <- tm_map(cleanCorpus, toSpace, "[[:punct:]]")
    cleanCorpus <- tm_map(cleanCorpus, toSpace, "[0-9]")
    
    ## removePunctuation converts interviews...oddly to interviewsoddly
    ## cleanCorpus <- tm_map(cleanCorpus, removeNumbers)
    ## cleanCorpus <- tm_map(cleanCorpus, removePunctuation)
    
    cleanCorpus <- tm_map(cleanCorpus, stripWhitespace)
    
    return(cleanCorpus)
}

## Convert training data to corpus and inspect a few documents
trainCorpus <- SimpleCorpus(VectorSource(rawTrainData))
inspect(trainCorpus[c(1,length(trainCorpus)%/%2, length(trainCorpus)-20)])

## Clean the training corpus
trainCorpus <- preprocessData(trainCorpus, isStopWordsRemoved=FALSE)
inspect(trainCorpus[c(1,length(trainCorpus)%/%2, length(trainCorpus)-20)])


###################
## Explore the data
###################

## getTermFrequency: get frequencies of terms in a corpus, in decreasing order.
##
## corpus = input a corpus
## sparse = input a sparse fraction 0 to 1
## freqDf = output a data frame of terms and frequencies
##
getTermFrequency <- function(corpus, sparse) {
    if (!require(tm)) {
        stop("Library tm is missing.")
    }
    ## Convert to DocumnetTermMatrix
    dtm <- DocumentTermMatrix(corpus)
    
    ## Get frequencies of terms
    if (sparse >= 1) {
        freq <- colSums(as.matrix(dtm))
    } else {
        freq <- colSums(as.matrix(removeSparseTerms(dtm, sparse=sparse)))
    }
    
    ## Sort in decreasing order and output
    freq <- sort(freq, decreasing=TRUE)
    freqDf <- data.frame(ngram=names(freq), freq=freq)
}

## getNgram: extracts ngrams with the ngram library
## corpus = input, a SimpleCorpus
## n = input, number of word in ngrams
## ng = output, a dataframe of ngram, freq, and probability
##
getNgram <- function(corpus, n=2) {
    if (!require(ngram)) {
        stop("Library ngram is missing.")
    }
    
    ## Convert corpus to a string
    str <- concatenate(corpus$content)
    ng <- ngram(str, n=n)
    return(get.phrasetable(ng))
}

## summaryNgram: summarizes the ngram generated by getNgram
## ngram = input, a dataframe from getNgram
## summaryTable = output, a dataframe of number of ngrams 
##   at frequencies >= 1, 2, 3, and 4
##
summaryNgram <- function(ngram) {
    freqNWords <- function(n) sum(ngram$freq >= n)
    freqNOccurence <- function(n) sum(ngram[ngram$freq >= n, "freq"])
    n2 = 10
    n3 = 25
    n4 = 50
    n5 = 100
    totalWords <- dim(ngram)[1]
    freq2Words <- freqNWords(n2)
    freq3Words <- freqNWords(n3)
    freq4Words <- freqNWords(n4)
    freq5Words <- freqNWords(n5)
    numWords <- c(totalWords, freq2Words, freq3Words, freq4Words, freq5Words)
    fracWords <- c(1.0, round(freq2Words/totalWords, 4),
                   round(freq3Words/totalWords, 4), 
                   round(freq4Words/totalWords, 4),
                   round(freq5Words/totalWords, 4))
    totalOccurrence <- sum(ngram$freq)
    freq2Occurrence <- freqNOccurence(n2)
    freq3Occurrence <- freqNOccurence(n3)
    freq4Occurrence <- freqNOccurence(n4)
    freq5Occurrence <- freqNOccurence(n5)
    probWords <- c(1.0, round(freq2Occurrence/totalOccurrence, 4),
                   round(freq3Occurrence/totalOccurrence, 4),
                   round(freq4Occurrence/totalOccurrence, 4),
                   round(freq5Occurrence/totalOccurrence, 4))
    summaryData <- data.frame(numWords, fracWords, probWords)
    colnames(summaryData) <- c("Number_of_Words", "Fraction_of_Total", "Probability")
    rownames(summaryData) <- c("All", 
                               paste("Freq >= ", as.character(n2), collapse=''),
                               paste("Freq >= ", as.character(n3), collapse=''),
                               paste("Freq >= ", as.character(n4), collapse=''),
                               paste("Freq >= ", as.character(n5), collapse=''))
    
    return(summaryData)
}


# startTime <- proc.time()
# train1Gram <- getTermFrequency(trainCorpus, sparse=0.999)
# stopTime <- proc.time()
# print("Elapsed time of getTermFrequency 1 gram")
# stopTime - startTime
# 
# wordcloud(train1Gram$ngram, train1Gram$freq, scale=c(5,1), 
#           max.words=20, random.order=FALSE)

## Find the unigram
startTime <- proc.time()
train1Gram <- getNgram(trainCorpus, n=1)
stopTime <- proc.time()
print("Elapsed time of getNgram 1 gram")
stopTime - startTime

summaryNgram(train1Gram)
wordcloud(train1Gram$ngram, train1Gram$freq, scale=c(7, 1), 
          max.words=20, random.order=FALSE)

## Find the bigrams
startTime <- proc.time()
train2Gram <- getNgram(trainCorpus, n=2)
stopTime <- proc.time()
print("Elapsed time of getNgram 2 gram")
stopTime - startTime

summaryNgram(train2Gram)
wordcloud(train2Gram$ngram, train2Gram$freq, scale=c(3.5, 1), 
          max.words=20, random.order=FALSE)

## Find the trigrams
startTime <- proc.time()
train3Gram <- getNgram(trainCorpus, n=3)
stopTime <- proc.time()
print("Elapsed time of getNgram 3 gram")
stopTime - startTime

summaryNgram(train3Gram)
wordcloud(train3Gram$ngram, train3Gram$freq, scale=c(2.6, 1), 
          max.words=20, random.order=FALSE)

## getTopNFractions: gets fractions of top N ngrams in sets of 1-grams, 2-grams, 
## and 3-grams.
## ngram1 = input, set of 1-grams
## ngram2 = input, set of 2-grams
## ngram3 = input, set of 3-grams
## n = input, number of top ngrams
## topN = output, a vector of fractions of top N grams in the three sets
##
getTopNFractions <- function(ngram1, ngram2, ngram3, n=20) {
    topN.1gram <- round(sum(ngram1[1:n, "freq"])/sum(ngram1$freq), 4)
    topN.2gram <- round(sum(ngram2[1:n, "freq"])/sum(ngram2$freq), 4)
    topN.3gram <- round(sum(ngram3[1:n, "freq"])/sum(ngram3$freq), 4)
    topN <- c(topN.1gram, topN.2gram, topN.3gram)
    return(topN)
}

top20 <- getTopNFractions(train1Gram, train2Gram, train3Gram, n=20)
top100 <- getTopNFractions(train1Gram, train2Gram, train3Gram, n=100)
top10000 <- getTopNFractions(train1Gram, train2Gram, train3Gram, n=10000)
top50000 <- getTopNFractions(train1Gram, train2Gram, train3Gram, n=50000)
top100000 <- getTopNFractions(train1Gram, train2Gram, train3Gram, n=100000)

topSummary <- data.frame(top20, top100, top10000, top50000, top100000)
rownames(topSummary) <- c("Fraction of 1-grams", 
                          "Fraction of 2-grams", "Fraction of 3-grams")
topSummary


# 
# # cleanFreqDf <- data.frame(word=names(cleanFreq), freq=cleanFreq)
# # cleanStopWordsFreqDf <- data.frame(word=names(cleanStopWordsFreq), freq=cleanStopWordsFreq)
# 
# ## Draw the word cloud
# ## wordcloud(names(freq), freq, scale=c(5,1), max.words=20, random freq <- sort(colSums(as.matrix(cleanDtm)), decreasing=TRUE).order=FALSE)
# pal = brewer.pal(5, "GnBu")
# wordcloud(cleanCorpus, max.words=20, random.order=FALSE, scale=c(5, 1), colors=pal)
# wordcloud(cleanStopWordsCorpus, max.words = 20, random.order=FALSE, scale=c(3, 1), colors=pal)
# 
# wordcloud(cleanCorpus, max.words=20, random.order=FALSE, scale=c(6, 1))
# wordcloud(cleanStopWordsCorpus, max.words = 20, random.order=FALSE, scale=c(3, 1))
# 
# 
# ## Get bigrams
# 
# ## BirgramTokenizer function
# ## from hack-r.com/n-gram-wordclouds-in-r/
# BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min=2, max=2))
# 
# bgCleanDtm <- DocumentTermMatrix(cleanCorpus, control=list(tokenize=BigramTokenizer))
# 
# bgSparseCleanDtm <- removeSparseTerms(bgCleanDtm, sparse=sparseMax)
# 
# bgSparseFreq <- sort(colSums(as.matrix(bgSparseCleanDtm)), decreasing=TRUE)
# bgSparseFreqDf <- data.frame(word=names(bgSparseFreq), freq=bgSparseFreq)
# 
# wordcloud(words=bgSparseFreqDf$word, freq=bgSparseFreqDf$freq,
#           max.words=20, random.order=FALSE, scale=c(4, 1))
# 
# 
# ## Get trigrams
# TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min=3, max=3))
# 
# tgCleanDtm <- DocumentTermMatrix(cleanCorpus, control=list(tokenize=TrigramTokenizer))
# 
# tgSparseCleanDtm <- removeSparseTerms(tgCleanDtm, sparse=0.9999)
# 
# tgSparseFreq <- sort(colSums(as.matrix(tgSparseCleanDtm)), decreasing=TRUE)
# tgSparseFreqDf <- data.frame(word=names(tgSparseFreq), freq=tgSparseFreq)
# 
# wordcloud(words=tgSparseFreqDf$word, freq=tgSparseFreqDf$freq,
#           max.words=20, random.order=FALSE, scale=c(3, 1))






## gsub("[^[:alnum:]]", " ",x)
## gsub("[<>\\'\"]", " ", "Hi <bob> that's Alice there.")


