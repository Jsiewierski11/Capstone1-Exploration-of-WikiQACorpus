from cleaning import *


def make_graphs_pipeline(df, cats, remove_stop=False):
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

if __name__ == '__main__':
    # reading in data
    text_df = pd.read_csv('data/WikiQA.tsv', sep="\t")

    # reading in columns to select/drop
    titles = read_cols('data/titles_to_use.txt')
    drops = read_cols('data/cols_to_drop.txt')

    # Narrowing dataset    
    healthcare_df = select_doc_titles(text_df, titles)
    healthcare_df = drop_cols(healthcare_df, drops)

    # Bar chart of answers per question
    val_count = healthcare_df['DocumentTitle'].value_counts()
    make_barchart(val_count.index, val_count, filepath='graphs/answers_per_question.png', figsize=(35, 15), title='Number of answers for each category')    
    
    # Word Cloud of Top 5 Categories with the most answers
    top_6 = ['List of muscles of the human body', 'Meiosis', 'Mandibular first molar', \
             'Comparison of the health care systems in Canada and the United States', \
             'Antibody', 'Cellular respiration']

    wordcloud_of_answers(healthcare_df, top_6)

    counts_df = count_words_pipeline(healthcare_df)
    total_counts_df = count_words_pipeline(text_df)
    pretty_plot_top_n(counts_df['n_w'], top_n=3, filepath='graphs/wordcount_with_stopwords.png')
    top_n_dfs = make_graphs_pipeline(healthcare_df, top_6)

    plot_multi_top_n(top_n_dfs, filepath='graphs/collaged_topn.png', top_n=5, numrows=2, numcols=3)
    make_counts_violin(counts_df, filepath='graphs/healthcare_violin.png')
    make_counts_violin(total_counts_df, filepath='graphs/wikiQA_violin.png')

    counts_df = count_words_pipeline(healthcare_df, remove_stop=True)
    total_counts_df = count_words_pipeline(text_df, remove_stop=True)
    pretty_plot_top_n(counts_df['n_w'], top_n=3, filepath='graphs/wordcount_no_stopwords.png')
    top_n_dfs = make_graphs_pipeline(healthcare_df, top_6, remove_stop=True)
    plot_multi_top_n(top_n_dfs, filepath='graphs/collaged_topn_no_stopwords.png', top_n=5, numrows=2, numcols=3)
    make_counts_violin(counts_df, filepath='graphs/healthcare_violin_no_stopwords.png')
    make_counts_violin(total_counts_df, filepath='graphs/wikiQA_violin_no_stopwords.png')

    plot_wordcloud(healthcare_df['DocumentTitle'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Question Types', \
                                 filepath='graphs/healthcareQs_no_stopwords_wc.png')

    plot_wordcloud(healthcare_df['Sentence'], max_words = 999999999999, title='Word Cloud of QA Healtcare Dataset Answers', \
                                 filepath='graphs/healthcareAs_no_stopwords_wc.png')

    next_6 = ['Vitamin', 'Diagnosis-related group', 'Multiple sclerosis', 'Vitamin A', 'Digestion', \
              'Health care in the United States']
    next_n_dfs = make_graphs_pipeline(healthcare_df, next_6)
    plot_multi_top_n(next_n_dfs, filepath='graphs/collaged_nextn.png', top_n=5, numrows=2, numcols=3)

    next_n_dfs = make_graphs_pipeline(healthcare_df, next_6, remove_stop=True)
    plot_multi_top_n(next_n_dfs, filepath='graphs/collaged_nextn_no_stopwords.png', top_n=5, numrows=2, numcols=3)