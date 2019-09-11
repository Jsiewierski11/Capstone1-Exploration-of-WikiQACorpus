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
    for row in df[['Question', 'words']].iterrows():
        r = row[1]
        for word in r.words:
            rows.append((r['Question'], word))

    return pd.DataFrame(rows, columns=['Question', 'word'])

def remove_empty_word_cols(df):
    words = df[df.word.str.len() > 0]
    words.word.str.lower()
    return words

def count_word_ocurrence(df):
    counts = df.groupby('Question')\
            .word.value_counts()\
            .to_frame()\
            .rename(columns={'word':'n_w'})
    return counts


def count_words_pipeline(df):
    df = create_words_col(df)
    words_df = parse_words_to_word(df)
    words_df = remove_empty_word_cols(words_df)
    return count_word_ocurrence(words_df)

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

    # Bar chart of answers per question
    val_count = healthcare_df['DocumentTitle'].value_counts()
    make_barchart(val_count.index, val_count, filepath='graphs/answers_per_question.png', figsize=(35, 15), title='Number of answers for each category')    
    
    # Word Cloud of Top 5 Categories with the most answers
    top_5 = ['List of muscles of the human body', 'Meiosis', 'Mandibular first molar', 'Drug test', 'Antibody', 'Cellular respiration']
    wordcloud_of_answers(healthcare_df, top_5)

    counts_df = count_words_pipeline(healthcare_df)
    pretty_plot_top_n(counts_df['n_w'], top_n=1)

    top_n_dfs = []
    for label in top_5:
        temp_df = get_all_answers(healthcare_df, label)
        cleaned = count_words_pipeline(temp_df)
        top_n_dfs.append(cleaned['n_w'])

    plot_multi_top_n(top_n_dfs, filepath='graphs/collaged_topn.png', numrows=2, numcols=3)