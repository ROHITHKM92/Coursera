<div class="container-fluid main-container">

<div class="row">

<div class="col-xs-12">

<div id="TOC" class="tocify">

</div>

</div>

<div class="toc-content col-xs-12">

<div id="header">

# <span style="color:#2E6AA8;">Data Science Specialization Capstone: Milestone Report</span>

### <span style="font-size:small; font-style:italic;">Peer-graded Assignment - Data Science Specialization Capstone Milestone Report - Johns Hopkins University</span>

#### Rohith Mohan

#### April 28, 2024

</div>

<div id="introduction" class="section level1">

# Introduction

Welcome to the Week 2 Peer-graded assignment for the Johns Hopkins Data
Science Capstone course. This report serves as a comprehensive overview
of the tasks undertaken in this assignment. Throughout this document, I
aim to:

1.  Validate the successful download and loading of the data into R.
2.  Provide a detailed summary statistics report on the datasets.
3.  Highlight any intriguing discoveries made during the analysis.
4.  Solicit feedback on proposed strategies for developing a prediction
    algorithm and a Shiny application.

Loading required libraries.

``` r
library(ggplot2)
library(dplyr)
library(stringi)
library(tm)
library(wordcloud)
library(RWeka)
library(tm)
library(pryr)
library(RColorBrewer)
library(viridis)
library(hunspell)
```

</div>

<div id="download-and-extract-the-data" class="section level1">

# Download and extract the Data

``` r
if (!file.exists("Coursera-SwiftKey.zip")){
        download.file(url = "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip", destfile = "Coursera-SwiftKey.zip")
        unzip("Coursera-SwiftKey.zip")
}
```

</div>

<div id="basic-summary" class="section level1">

# Basic Summary

Let’s explore the contents of the dataset:

``` r
 list.files("final")
```

    ## [1] "de_DE" "en_US" "fi_FI" "ru_RU"

We possess data for four languages: German, English, Finnish, and
Russian. For this project, we exclusively utilize the English-language
data. Let’s examine the files present within the English-language
dataset: We consider only the Enlish language files.

``` r
list.files("final/en_US")
```

    ## [1] "en_US.blogs.txt"   "en_US.news.txt"    "en_US.twitter.txt"

We’ve acquired English-language data from three distinct sources:

1.  Blogs (en_US.blogs.txt)
2.  News (en_US.news.txt)
3.  Twitter (en_US.twitter.txt)

Following this, the data will be imported into R for analysis.

``` r
blogs <- readLines("final/en_US/en_US.blogs.txt", warn = FALSE, encoding = "UTF-8", skipNul = TRUE) # Blogs

news <- readLines("final/en_US/en_US.news.txt", warn = FALSE, encoding = "UTF-8", skipNul = TRUE) # News

twitter <- readLines("final/en_US/en_US.twitter.txt", warn = FALSE, encoding = "UTF-8", skipNul = TRUE) # Twitter
```

``` r
Summary_Stats <- data.frame(
  FileName = c("blogs", "news", "twitter"),
  FileSize = sapply(list(blogs, news, twitter), function(x) format(object.size(x), "MB")),
  t(rbind(
    sapply(list(blogs, news, twitter), stri_stats_general)[c("Lines", "Chars"),], 
    Words = sapply(list(blogs, news, twitter), stri_stats_latex)[4,]
  ))
)

Summary_Stats
```

    ##   FileName FileSize   Lines     Chars    Words
    ## 1    blogs 255.4 Mb  899288 206824382 37570839
    ## 2     news  19.8 Mb   77259  15639408  2651432
    ## 3  twitter   319 Mb 2360148 162096241 30451170

</div>

<div id="sampling-and-aggregating-data" class="section level1">

# Sampling and Aggregating Data

Upon reviewing the summary, it’s evident that the data files’ sizes are
notably large. To address this, we plan to subset the data into three
new files, each containing a 2% sample of the original data files.
Initially, we’ll begin with a 2% sample and assess the size of the
VCorpus (Virtual Corpus) object that will be loaded into memory.

For reproducibility, we’ll set a seed to ensure consistent sampling.
Prior to constructing the corpus, we’ll create a combined sample file.
Subsequently, we’ll reevaluate the summary statistics to verify that the
file sizes remain manageable.

``` r
set.seed(66666)
sampleSize <- 0.02

# Create samples
blogs_Sample <- sample(blogs, length(blogs) * sampleSize)
news_Sample <- sample(news, length(news) * sampleSize)
twitter_Sample <- sample(twitter, length(twitter) * sampleSize)
sampleData <- c(blogs_Sample, news_Sample, twitter_Sample)

# Summary statistics for sample data
sampleStats <- data.frame(
  FileName = c("blogs_Sample", "news_Sample", "twitter_Sample", "sampleData"),
  FileSize = sapply(list(blogs_Sample, news_Sample, twitter_Sample, sampleData), function(x) format(object.size(x), "MB")),
  Lines = sapply(list(blogs_Sample, news_Sample, twitter_Sample, sampleData), function(x) length(x)),
  Chars = sapply(list(blogs_Sample, news_Sample, twitter_Sample, sampleData), function(x) sum(nchar(x))),
  Words = sapply(list(blogs_Sample, news_Sample, twitter_Sample, sampleData), function(x) sum(sapply(x, function(y) length(unlist(strsplit(y, "\\s+"))))))
)
sampleStats
```

    ##         FileName FileSize Lines   Chars   Words
    ## 1   blogs_Sample   5.1 Mb 17985 4136931  747832
    ## 2    news_Sample   0.4 Mb  1545  322150   54568
    ## 3 twitter_Sample   6.5 Mb 47202 3249502  608930
    ## 4     sampleData    12 Mb 66732 7708583 1411330

Construct a Corpus

``` r
corp <- VCorpus(VectorSource(sampleData))  # Build the corpus

# Check the size of the corpus in memory using the object_size function from the pryr package.
pryr::object_size(corp)
```

    ## 155.63 MB

Even at a 2% sample size, the VCorpus object still occupies a
significant amount of memory, totaling 155.63 MB. This size may present
challenges due to memory constraints, especially when constructing
predictive models. However, we will continue with this corpus and
monitor its impact as we move forward.

</div>

<div id="data-cleaning" class="section level1">

# Data Cleaning

Our next step involves cleaning the corpus data using functions from the
tm package. Typical text mining cleaning tasks encompass:

1.  Converting all words to lowercase.
2.  Removing all white spaces.
3.  Eliminating punctuation.
4.  Removing numerical digits.
5.  Stripping away various non-alphanumeric characters.
6.  Eliminating stop words (commonly occurring but uninformative words
    such as “the”, “and”, “also”, etc.).
7.  Removing URLs.
8.  Filtering out profanity.

``` r
# Define function to preprocess text
preprocess_text <- function(text) {
  # Convert to lowercase
  text <- tolower(text)
  # Remove punctuation marks
  text <- removePunctuation(text)
  # Remove numbers
  text <- removeNumbers(text)
  # Remove whitespace
  text <- stripWhitespace(text)
  # Convert to plain text document
  text <- PlainTextDocument(text)
  return(text)
}


removeURL <- function(x) gsub("http[[:alnum:][:punct:]]*", "", x) # Define function to remove URLs

# Remove profanities
# The list used can be found here: https://www.cs.cmu.edu/~biglou/resources/bad-words.txt
profanity_lines <- readLines("badwords.txt") # Read the profanity list from the file


profanities <- trimws(profanity_lines) # Remove any leading or trailing whitespace from the profanity words

remove_special_chars <- content_transformer(function(x, pattern) gsub(pattern, " ", x)) # Define function to remove special characters

# Perform transformations
corp <- tm_map(corp, content_transformer(preprocess_text)) # Preprocess text
corp <- tm_map(corp, remove_special_chars, "#|/|@|\\|")    # Remove special characters
corp <- tm_map(corp, removeWords, stopwords("english"))    # Remove stopwords
corp <- tm_map(corp, content_transformer(removeURL))      # Remove URLs
corp <- tm_map(corp, removeWords, profanities)            # Remove profanities
```

</div>

<div id="tokenization-and-n-gram-construction" class="section level1">

# Tokenization and N-Gram Construction:

Our next task involves tokenizing the cleaned corpus, which entails
breaking the text into individual words and short phrases, to construct
a set of N-grams. We’ll begin with three types of N-grams:

1.  Unigram: A matrix comprising individual words.
2.  Bigram: A matrix containing two-word patterns.
3.  Trigram: A matrix containing three-word patterns.

While we could also construct a Quadgram matrix based on four words,
we’ve chosen to prioritize the first three N-grams for now. We’ll assess
the performance of the predictive model using these N-grams before
considering additional complexities.

``` r
# Define tokenization functions for unigrams, bigrams, and trigrams
uniTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
biTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
triTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))

uniMatrix <- TermDocumentMatrix(corp, control = list(tokenize = uniTokenizer))
biMatrix <- TermDocumentMatrix(corp, control = list(tokenize = biTokenizer))
triMatrix <- TermDocumentMatrix(corp, control = list(tokenize = triTokenizer))
```

</div>

<div id="compute-the-frequencies-of-n-grams-and-visualize"
class="section level1">

# Compute the Frequencies of N-Grams and Visualize

Next, we’ll compute the frequencies of the N-Grams, examine their
distributions, and illustrate the data by generating visual
representations of the dataset.

<div id="unigrams" class="section level2">

## Unigrams

``` r
# Find frequent terms for unigrams
uniCorpus <- findFreqTerms(uniMatrix, lowfreq = 4)
```

Calculate frequencies for unigrams

``` r
uniCorpusFreq <- rowSums(as.matrix(uniMatrix[uniCorpus,]))
uniCorpusFreq <- data.frame(word = names(uniCorpusFreq), frequency = uniCorpusFreq)
uniCorpusFreq <- arrange(uniCorpusFreq, desc(frequency))
head(uniCorpusFreq)
```

    ##      word frequency
    ## just just      5067
    ## like like      4494
    ## will will      4418
    ## one   one      4139
    ## get   get      3732
    ## can   can      3731

Here’s a word cloud visualizing the frequencies of the most common words

``` r
wordcloud(words = uniCorpusFreq$word,
          freq = uniCorpusFreq$frequency,
          min.freq = 1,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(9, "Set1"))
```

![UW](https://github.com/ROHITHKM92/Coursera/assets/87298902/640a0da5-f530-4954-ad11-a18808e4478b)<!-- -->

The below plot displays the frequency distribution of the top 15 most
common unigrams, indicating the occurrence of each unigram in the
corpus.

``` r
uniCorpusFreq <- uniCorpusFreq[order(-uniCorpusFreq$frequency), ]

unigram_hist <- ggplot(uniCorpusFreq[1:15, ], aes(x = frequency, y = reorder(word, frequency), fill = -frequency))  # Reverse order of fill aesthetic
unigram_hist <- unigram_hist + geom_col(width = 0.5) +
  scale_fill_viridis(option = "magma") +  
  labs(x = "Frequency", y = "Unigram", title = "15 Most Frequent Unigrams", fill = "Frequency") +  # Set legend title
  theme_minimal() +
  theme(plot.title = element_text(size = 14, hjust = 0.5),
        axis.text.x = element_text(angle = 0),
        axis.text.y = element_text(hjust = 1))
print(unigram_hist)
```

![UB](https://github.com/ROHITHKM92/Coursera/assets/87298902/834f9524-1cb7-4bfd-b5ea-d707be3ceb61)<!-- -->


<div id="word-frequency-and-english-language-corpus-analysis"
class="section level3">

### Word Frequency and English Language Corpus Analysis

The prevalence of the most frequently used words indicates that a small
portion of the total unique words make up the majority of the corpus.
We’ll delve into determining the quantity of unique words required to
encapsulate 50% and 90% of all occurrences within the language
represented by the corpus. It’s important to note that this analysis
excludes stopwords, and the overall count of words needed to achieve the
specified thresholds would notably decrease upon their inclusion.

``` r
# Calculate the cumulative percentage of word frequencies
uniCorpusFreq$cum <- cumsum(uniCorpusFreq$frequency) / sum(uniCorpusFreq$frequency)

# Determine the number of words needed to cover 50% of the corpus
num_words_50_percent <- which(uniCorpusFreq$cum >= 0.5)[1]
print(paste("Number of words needed to cover 50% of the corpus = ", num_words_50_percent, sep = ""))
```

    ## [1] "Number of words needed to cover 50% of the corpus = 579"

``` r
# Determine the number of words needed to cover 90% of the corpus
num_words_90_percent <- which(uniCorpusFreq$cum >= 0.9)[1]
print(paste("Number of words needed to cover 90% of the corpus = ", num_words_90_percent, sep = ""))
```

    ## [1] "Number of words needed to cover 90% of the corpus = 6450"

The subsequent stage involves assessing the proportion of
English-language words within the Corpus. This estimation process
comprises the following steps:

1.  Tallying the total number of words in the corpus.
2.  Employing an English-language spellchecker (hunspell_check) to
    validate words against the corresponding dictionary and eliminating
    those that do not match.
3.  Counting the remaining number of words in the corpus.

``` r
# Define a function to perform English-language spellchecking
english_spellcheck <- function(words) {
  english_words <- hunspell_check(words)
  return(english_words)
}

# Print the total number of words in the corpus before applying the spellchecker
total_words_before_spellcheck <- nrow(uniCorpusFreq)

print(paste("The total number of unique words in the corpus (before applying the spellchecker) = ", total_words_before_spellcheck, sep = ""))
```

    ## [1] "The total number of unique words in the corpus (before applying the spellchecker) = 16138"

``` r
# Apply English-language spellchecking to unigram words
uniCorpusFreq$english <- english_spellcheck(uniCorpusFreq$word)

# Filter out words not found in the English dictionary
uniCorpusFreq_ed <- uniCorpusFreq[uniCorpusFreq$english, ]

# Count the number of remaining English-language words
num_words_remaining <- nrow(uniCorpusFreq_ed)
print(paste("The total number of English-language words in the corpus (after applying the spellchecker) = ", num_words_remaining, sep = ""))
```

    ## [1] "The total number of English-language words in the corpus (after applying the spellchecker) = 13194"

As misspelled words might frequently include names or specialized terms
not found in a standard English dictionary, the spellchecker will not be
utilized for the remainder of the analysis on the corpus.

</div>

</div>

<div id="bigrams" class="section level2">

## Bigrams

``` r
# Find frequent terms for bigrams
biCorpus <- findFreqTerms(biMatrix, lowfreq = 4)
```

Calculate frequencies for bigrams

``` r
biCorpusFreq <- rowSums(as.matrix(biMatrix[biCorpus,]))
biCorpusFreq <- data.frame(word = names(biCorpusFreq), frequency = biCorpusFreq)
biCorpusFreq <- arrange(biCorpusFreq, desc(frequency))
head(biCorpusFreq)
```

    ##                            word frequency
    ## right now             right now       465
    ## cant wait             cant wait       394
    ## last night           last night       311
    ## dont know             dont know       260
    ## feel like             feel like       230
    ## looking forward looking forward       227

Here’s a word cloud showcasing the frequencies of the most common
bigrams.

``` r
wordcloud(words = biCorpusFreq$word,
          freq = biCorpusFreq$frequency,
          min.freq = 1,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(9, "Set1"))
```

![BW](https://github.com/ROHITHKM92/Coursera/assets/87298902/3e0b35a0-5241-45c1-b990-6864a8220fd5)<!-- -->

The histogram below depicts the frequencies of the top 15 most common
words.

``` r
biCorpusFreq <- biCorpusFreq[order(-biCorpusFreq$frequency), ]

bigrams_hist <- ggplot(biCorpusFreq[1:15, ], aes(x = frequency, y = reorder(word, frequency), fill = -frequency))  # Reverse order of fill aesthetic
bigrams_hist <- bigrams_hist + geom_col(width = 0.5) +
  scale_fill_viridis(option = "magma") +  
  labs(x = "Frequency", y = "Bigrams", title = "15 Most Frequent Bigrams", fill = "Frequency") +  # Set legend title
  theme_minimal() +
  theme(plot.title = element_text(size = 14, hjust = 0.5),
        axis.text.x = element_text(angle = 0),
        axis.text.y = element_text(hjust = 1))
print(bigrams_hist)
```

![BB](https://github.com/ROHITHKM92/Coursera/assets/87298902/94c6fae9-ad4e-46ab-9888-c59838c08e01)<!-- -->

</div>

<div id="trigrams" class="section level2">

## Trigrams

``` r
# Find frequent terms for trigrams
triCorpus <- findFreqTerms(triMatrix, lowfreq = 2)
```

Calculate frequencies for trigrams

``` r
triCorpusFreq <- rowSums(as.matrix(triMatrix[triCorpus,]))
triCorpusFreq <- data.frame(word = names(triCorpusFreq), frequency = triCorpusFreq)
triCorpusFreq <- arrange(triCorpusFreq, desc(frequency))
head(triCorpusFreq)
```

    ##                                word frequency
    ## cant wait see         cant wait see        75
    ## happy mothers day happy mothers day        61
    ## let us know             let us know        48
    ## happy new year       happy new year        41
    ## cinco de mayo         cinco de mayo        31
    ## im pretty sure       im pretty sure        27

Here, we visualize a word cloud illustrating the frequencies of the most
prevalent trigrams.

``` r
wordcloud(words = triCorpusFreq$word,
          freq = triCorpusFreq$frequency,
          min.freq = 1,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(9, "Set1"))
```

![TW](https://github.com/ROHITHKM92/Coursera/assets/87298902/a0b27aea-42fd-451c-8ae7-95c019d17041)<!-- -->


The histogram provided below illustrates the frequencies of the top 15
most common words.

``` r
triCorpusFreq <- triCorpusFreq[order(-triCorpusFreq$frequency), ]

trigrams_hist <- ggplot(triCorpusFreq[1:15, ], aes(x = frequency, y = reorder(word, frequency), fill = -frequency))  # Reverse order of fill aesthetic
trigrams_hist <- trigrams_hist + geom_col(width = 0.5) +
  scale_fill_viridis(option = "magma") +  
  labs(x = "Frequency", y = "Trigrams", title = "15 Most Frequent Trigrams", fill = "Frequency") +  # Set legend title
  theme_minimal() +
  theme(plot.title = element_text(size = 14, hjust = 0.5),
        axis.text.x = element_text(angle = 0),
        axis.text.y = element_text(hjust = 1))
print(trigrams_hist)
```

![TB](https://github.com/ROHITHKM92/Coursera/assets/87298902/e633d8ac-572c-4bb0-a9a3-825b0ff177f8)<!-- -->

</div>

</div>

<div id="saving-essential-data" class="section level1">

# Saving Essential Data

We store the collections of unigrams, bigrams, and trigrams onto disk,
essential for their utilization in the subsequent phase, the development
of the predictive model.

``` r
saveRDS(uniCorpusFreq, file = "unigrams_data.rds")
saveRDS(biCorpusFreq, file = "bigrams_data.rds")
saveRDS(triCorpusFreq, file = "trigrams_data.rds")
```

</div>

<div id="next-steps-in-developing-the-predictive-model"
class="section level1">

# Next Steps in Developing the Predictive Model:

Moving forward, our predictive model will utilize a combination of
unigrams, bigrams, and trigrams to forecast the next word based on input
text. These n-gram models can be effectively stored as Markov chains,
streamlining model complexity. Markov chains capture the probabilities
of transitioning to another state given the current state, mirroring the
likelihood of certain words occurring after a unigram, bigram, or
trigram.

To determine predicted words, we’ll employ backoff models. One method
involves initially assessing the probability distribution of the next
word given a trigram. Should the trigram not exist, we’ll then consider
the probability distribution from the bigram, and similarly the unigram
model if necessary. However, due to limited data within the corpus,
predictions based on a specific trigram or bigram may be overly
constrained or focused on a singular topic. To address this, we’ll
assign relative weights to trigrams, bigrams, and unigrams within the
predictive model. For instance, the “Stupid Backoff” technique allocates
scores to each n-gram category based on their relative frequencies.

An essential consideration is how to expand the coverage of the English
language, either by identifying words absent from the corpus or
utilizing fewer words in the dictionary to encompass the same number of
phrases. With current memory constraints, our corpus comprises only
16138 words, potentially resulting in user inputs not recognized within
the corpus. To mitigate this, we’ll categorize a small percentage of
words with minimal occurrences as “unknown.” We’ll then calculate the
probability distribution of words following these “unknown” words and
apply this distribution to words absent from the corpus. This
classification simplifies the model by reducing the total unique number
of words within the corpus.

Additionally, synonyms will be treated as the same word in the
predictive model, potentially leading to new user words being classified
as existing corpus words.

The final model will be deployed as a Shiny application.

</div>

</div>

</div>

</div>
