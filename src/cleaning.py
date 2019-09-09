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
    titles = read_cols('data/cols_to_use.txt')
    drops = read_cols('data/cols_to_drop.txt')

    # cleaning data
    small_df = select_doc_titles(text_df, titles)
    small_df = drop_cols(small_df, drops)
    # print(small_df)

    jefferson_df = small_df[small_df['DocumentTitle'] == 'Thomas Jefferson'].reset_index()
    plot_wordcloud(jefferson_df['Question'], title='Word Cloud of Questions relating to Thomas Jefferson', \
                   filepath='graphs/jefferson_question_wc.png')

    plot_wordcloud(jefferson_df['Sentence'], title='Word Cloud of Questions relating to Thomas Jefferson', \
                   filepath='graphs/jefferson_sentence_wc.png')