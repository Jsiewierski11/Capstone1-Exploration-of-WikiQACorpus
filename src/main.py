from cleaning import *

def make_violins_pipeline(df, cats, remove_stop=False):
        df_lst = []
        for label in cats:
            temp_df = get_all_answers(df, label)
            cleaned = count_words_pipeline(temp_df, remove_stop=remove_stop)
            if remove_stop:
                make_counts_violin(cleaned, filepath='graphs/{}_violin_no_stop.png'.format(label))
            else:
                make_counts_violin(cleaned, filepath='graphs/{}_violin.png'.format(label))
            df_lst.append(cleaned['n_w'])
        return df_lst

def count_words_in_data(healthcare_df, text_df, remove_stop=False):
    counts_df = count_words_pipeline(healthcare_df, remove_stop=remove_stop)
    total_counts_df = count_words_pipeline(text_df, remove_stop=remove_stop)
    return counts_df, total_counts_df

def plotting_word_counts(healthcare_df, text_df, counts_df, total_counts_df, labels):
    pretty_plot_top_n(counts_df['n_w'], top_n=3, filepath='graphs/wordcount_with_stopwords.png')
    # pretty_plot_top_n(total_counts_df['n_w'], top_n=3, filepath='graphs/wordcount_wikiQA_stopwords.png')
    top_n_dfs = make_violins_pipeline(healthcare_df, labels)
    plot_zipf(top_n_dfs, filepath='graphs/collaged_zipf.png', top_n=50, numrows=2, numcols=3)
    plot_multi_top_n(top_n_dfs, filepath='graphs/collaged_topn.png', top_n=10, numrows=2, numcols=3)
    make_counts_violin(counts_df, filepath='graphs/healthcare_violin.png')
    make_counts_violin(total_counts_df, filepath='graphs/wikiQA_violin.png')

def plotting_word_counts_no_stopwords(healthcare_df, text_df, counts_df, total_counts_df, labels):
    pretty_plot_top_n(counts_df['n_w'], top_n=3, filepath='graphs/wordcount_no_stopwords.png')
    # pretty_plot_top_n(total_counts_df['n_w'], top_n=3, filepath='graphs/wordcount_wiki_no_stopwords.png')
    top_n_dfs = make_violins_pipeline(healthcare_df, labels, remove_stop=True)
    plot_zipf(top_n_dfs, filepath='graphs/collaged_zipf_no_stopwords.png', top_n=50, numrows=2, numcols=3)
    plot_multi_top_n(top_n_dfs, filepath='graphs/collaged_topn_no_stopwords.png', top_n=10, numrows=2, numcols=3)
    make_counts_violin(counts_df, filepath='graphs/healthcare_violin_no_stopwords.png')
    make_counts_violin(total_counts_df, filepath='graphs/wikiQA_violin_no_stopwords.png')  

def word_cloud_whole_dataset(df, remove_stop=False):
    if remove_stop:
        plot_wordcloud(df['DocumentTitle'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Question Types', \
                                    filepath='graphs/healthcareQs_no_stopwords_wc.png')

        plot_wordcloud(df['Sentence'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Answers', \
                                 filepath='graphs/healthcareAs_no_stopwords_wc.png') 
    else:
        plot_wordcloud(df['DocumentTitle'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Question Types', \
                                    filepath='graphs/healthcareQs_wc.png')

        plot_wordcloud(df['Sentence'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Answers', \
                                 filepath='graphs/healthcareAs_wc.png') 

if __name__ == '__main__':
    # reading in data
    text_df = pd.read_csv('data/WikiQA.tsv', sep="\t")

    # reading in columns to select/drop
    titles = read_cols('data/titles_to_use.txt')
    drops = read_cols('data/cols_to_drop.txt')

    # Narrowing dataset    
    healthcare_df = select_doc_titles(text_df, titles)
    healthcare_df = drop_cols(healthcare_df, drops)

    val_count = healthcare_df['DocumentTitle'].value_counts()
    # make_barchart(val_count.index, val_count, filepath='graphs/answers_per_question.png', figsize=(35, 15), title='Number of answers for each category')    
    word_cloud_whole_dataset(healthcare_df)

    # val_count = text_df['DocumentTitle'].value_counts()
    # word_cloud_whole_dataset(text_df)

    counts_df, total_counts_df = count_words_in_data(healthcare_df, text_df)
    top_6 = ['List of muscles of the human body', 'Meiosis', 'Mandibular first molar', \
             'Comparison of the health care systems in Canada and the United States', \
             'Antibody', 'Cellular respiration']
    plotting_word_counts(healthcare_df, text_df, counts_df, total_counts_df, top_6)


    counts_df, total_counts_df = count_words_in_data(healthcare_df, text_df, remove_stop=True)
    plotting_word_counts_no_stopwords(healthcare_df, text_df, counts_df, total_counts_df, top_6)
    
    
    next_6 = ['Vitamin', 'Diagnosis-related group', 'Multiple sclerosis', 'Vitamin A', 'Digestion', \
              'Health care in the United States']
    next_n_dfs = make_violins_pipeline(healthcare_df, next_6)
    plot_zipf(next_n_dfs, filepath='graphs/next6_zipf.png', top_n=50, numrows=2, numcols=3)
    plot_multi_top_n(next_n_dfs, filepath='graphs/collaged_nextn.png', top_n=10, numrows=2, numcols=3)
    next_n_dfs = make_violins_pipeline(healthcare_df, next_6, remove_stop=True)
    plot_multi_top_n(next_n_dfs, filepath='graphs/collaged_nextn_no_stopwords.png', top_n=10, numrows=2, numcols=3)
    plot_zipf(next_n_dfs, filepath='graphs/next6_zipf_no_stopwords.png', top_n=50, numrows=2, numcols=3)