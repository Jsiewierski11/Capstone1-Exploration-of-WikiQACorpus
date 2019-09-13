
#import the wordcloud package
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from scipy import special

#Define the word cloud function with a max of 200 words
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False, filepath=None):
'''
Code below is taken from this Kaggle project:
https://www.kaggle.com/spurryag/beginner-attempt-at-nlp-workflow
That user sourced the code from here:
https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
'''
    stopwords = set(STOPWORDS)
    #define additional stop words that are not contained in the dictionary
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown', 'used'}
    stopwords = stopwords.union(more_stopwords)
    #Generate the word cloud
    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    #set the plot parameters
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def make_barchart(x, y, filepath=None, figsize=(12, 8), title=None, zipf=False):
    ax, fig = plt.subplots(1,1, figsize=figsize)
    plt.barh(range(len(x)), width=y)
    plt.yticks(range(len(x)), x, fontsize=14)
    if zipf:
        #define zipf distribution parameter. Has to be >1
        a = 2.
        end = len(x)
        new_x = np.array(range(1, end))
        y = (new_x)**(-a) / special.zetac(a)
        plt.plot(y/max(y), new_x, linewidth=2, color='r')

    if title is not None:
        fig.set_title(title, fontsize=25)
    
    plt.savefig(filepath)
    plt.close()

def pretty_plot_top_n(series, top_n=5, index_level=0, filepath=None):
    r = series\
        .groupby(level=index_level)\
        .nlargest(top_n)\
        .reset_index(level=index_level, drop=True)

    make_barchart(r.index, r, filepath=filepath, figsize=(50, 30), \
                  title='Count of the Most Frequent Word in all Answers for Each Question')

def plot_multi_top_n(series_arr, index_level=0, top_n=5, filepath=None, numrows=1, numcols=1):
    fig, axs = plt.subplots(numrows, numcols, figsize=(50,30))
    index = 0
    for i in range(0, numrows):
        for j in range(0, numcols):
            # Check for end of array
            if index == len(series_arr):
                plt.savefig(filepath)
                plt.tight_layout()
                plt.close()
                return
            r = series_arr[index] \
                .groupby(level=index_level) \
                .nlargest(top_n) \
                .reset_index(level=index_level, drop=True)
            ax = axs[i, j]
            ax.set_yticks(list(range(len(r.index))))
            ax.set_yticklabels(r.index)
            ax.barh(range(len(r.index)), width=r)

            title = r.index[0][0]
            ax.set_title(title)

            index += 1

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_zipf(series_arr, index_level=0, top_n=5, filepath=None, numrows=1, numcols=1):
    fig, axs = plt.subplots(numrows, numcols, figsize=(50,30))
    index = 0
    for i in range(0, numrows):
        for j in range(0, numcols):
            # Check for end of array
            if index == len(series_arr):
                plt.savefig(filepath)
                plt.tight_layout()
                plt.close()
                return
            r = series_arr[index] \
                .groupby(level=index_level) \
                .nlargest(top_n) \
                .reset_index(level=index_level, drop=True)
            ax = axs[i, j]
            ax.set_yticks(list(range(len(r.index))))
            ax.set_yticklabels(r.index)
            ax.barh(range(len(r.index)), width=np.log10(r))
            #define zipf distribution parameter. Has to be >1
            a = 1.0001
            end = len(r.index)
            new_x = np.array(range(1, end+1))
            y = (new_x)**(-a) / special.zetac(a)
            new_x = np.array(range(0, end))
            ax.plot(y/max(y), new_x, linewidth=2, color='r')

            title = r.index[0][0]
            ax.set_title(title)

            index += 1

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def make_counts_violin(df, filepath='graphs/violinplot.png'):
    fig, ax = plt.subplots(figsize =(9, 7)) 
    sns.violinplot(ax = ax,  y = df['n_w'] )
    fig.savefig(filepath)
    plt.close()