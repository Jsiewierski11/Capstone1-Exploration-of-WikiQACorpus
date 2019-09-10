import pandas as pd 
import numpy as np 
from plotting import *


def select_doc_titles(df, titles):
    '''
    Parameters:
        df - pandas dataframe from which we are selecting the columns from
        titles - an np array of document titles to select
    Returns:
        Pandas Dataframe of only desired document titles
    '''
    series = []
    for i in range(0, len(titles)):
        series.append(df[df['DocumentTitle'] == titles[i]].reset_index())

    return pd.concat(series).reset_index()


# NOTE: Make sure text file's last line is \n char
def read_cols(filepath):
    '''Reads text file of titles to read into array'''
    # define an empty list
    titles = []

    # open file and read the content in a list
    with open(filepath, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            titles.append(currentPlace)
    return titles

def drop_cols(df, cols):
    return df.drop(columns=cols)

if __name__ == '__main__':
    # reading in data
    text_df = pd.read_csv('data/WikiQA.tsv', sep="\t")

    # reading in columns to select/drop
    titles = read_cols('data/titles_to_use.txt')
    drops = read_cols('data/cols_to_drop.txt')

    # plot_wordcloud(text_df['DocumentTitle'], max_words=100000, title='Word Cloud of Whole Dataset\'s Doc Titles', \
    #                filepath='graphs/all_doc_titles.png')
    
    # plot_wordcloud(text_df['Sentence'], max_words=1000000000, title='Word Cloud of Whole Dataset\'s Sentences', \
                    # filepath='graphs/all_doc_sentences.png')

    # cleaning data
    
    healthcare_df = select_doc_titles(text_df, titles)
    healthcare_df = drop_cols(healthcare_df, drops)

    # plot_wordcloud(healthcare_df['DocumentTitle'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Question Types', \
    #                              filepath='graphs/healthcareQs_wc.png')

    # plot_wordcloud(healthcare_df['Sentence'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Answers', \
    #                              filepath='graphs/healthcareAs_wc.png')

    val_count = healthcare_df['DocumentTitle'].value_counts()
    make_barchart(val_count.index, val_count, filepath='graphs/answers_per_question.png', figsize=(35, 15), title='Number of answers for each category')    