import pandas as pd 
import numpy as np 
from plotting import *
from nltk.corpus import stopwords

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

def get_all_answers(df, category):
    return df[df['DocumentTitle'] == category].reset_index()

def wordcloud_of_answers(df, categories):
        for label in categories:
            new_df = get_all_answers(df, label)
            plot_wordcloud(new_df['Sentence'], max_words=1000, title=label, filepath='graphs/{}.png'.format(label))

def create_words_col(df):
    df['words'] = df['Sentence'].str.strip().str.split('[\W_]+')
    return df

def parse_words_to_word(df):
    rows = list()
    for row in df[['DocumentTitle', 'words']].iterrows():
        r = row[1]
        for word in r.words:
            rows.append((r['DocumentTitle'], word))

    return pd.DataFrame(rows, columns=['DocumentTitle', 'word'])

def remove_stop_words(df):
    stop = stopwords.words('english')
    df['word'] = df['word'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df

def remove_empty_word_cols(df):
    words = df[df.word.str.len() > 0]
    words.word.str.lower()
    return words

def count_word_ocurrence(df):
    counts = df.groupby('DocumentTitle')\
            .word.value_counts()\
            .to_frame()\
            .rename(columns={'word':'n_w'})
    return counts


def count_words_pipeline(df, remove_stop=False):
    df = create_words_col(df)
    words_df = parse_words_to_word(df)
    if remove_stop:
        words_df = remove_stop_words(words_df)
    words_df = remove_empty_word_cols(words_df)
    return count_word_ocurrence(words_df)

