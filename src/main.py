from cleaning import *


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

    # Narrowing dataset    
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
    top_5 = ['List of muscles of the human body', 'Meiosis', 'Mandibular first molar', \
             'Comparison of the health care systems in Canada and the United States', \
             'Antibody', 'Cellular respiration']

    wordcloud_of_answers(healthcare_df, top_5)

    counts_df = count_words_pipeline(healthcare_df)
    pretty_plot_top_n(counts_df['n_w'], top_n=1)

    top_n_dfs = []
    for label in top_5:
        temp_df = get_all_answers(healthcare_df, label)
        cleaned = count_words_pipeline(temp_df)
        top_n_dfs.append(cleaned['n_w'])

    # print(top_n_dfs[0])

    plot_multi_top_n(top_n_dfs, filepath='graphs/collaged_topn.png', top_n=5, numrows=2, numcols=3)