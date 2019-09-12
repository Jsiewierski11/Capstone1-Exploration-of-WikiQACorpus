'''
Code below is taken from this Kaggle project:
https://www.kaggle.com/spurryag/beginner-attempt-at-nlp-workflow
That user sourced the code from here:
https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
'''

#import the wordcloud package
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

#Define the word cloud function with a max of 200 words
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False, filepath=None):
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

def make_barchart(x, y, filepath=None, figsize=(12, 8), title=None):
    ax, fig = plt.subplots(1,1, figsize=figsize)
    plt.barh(range(len(x)), width=y)
    plt.yticks(range(len(x)), x, fontsize=14)

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

def make_counts_violin(df, filepath='graphs/violinplot.png'):
    fig, ax = plt.subplots(figsize =(9, 7)) 
    sns.violinplot(ax = ax,  y = df['n_w'] )
    fig.savefig(filepath)
    plt.close()


def pareto_plot(df, x=None, y=None, title=None, number_categories = 10, show_pct_y=False, pct_format='{0:.0%}'):    
    dfplot = df[[x,y]]
    #dfplot[y] = dfplot[y].abs() #some channels showed negative counts but I checked and they should be the positive value    dfsorted = dfplot.sort_values(y, ascending=False)    df_shortened = dfsorted[0:number_categories] #added for when there are too many categories to plot
    df_remaining = dfsorted[number_categories:df.shape[0]]    
    xlabel = x
    ylabel = y
    tmp = df_shortened.sort_values(y, ascending=False)
    tmp = tmp.append({x : 'Other' , y : df_remaining[y].abs().sum()}, ignore_index=True) #adds in an other category which has the sum of the remainder
    x = tmp[x].values
    y = tmp[y].values
    weights = y / y.sum()
    cumsum = weights.cumsum()    
    fig, ax1 = plt.subplots(figsize = (6,6)) #figsize adjusted to account for rotated labels
    ax1.bar(x, y)
    ax1.set_xlabel(xlabel)
    ax1.tick_params(axis = 'x', rotation = 90) #rotation for longer category names
    ax1.set_ylabel(ylabel)    
    ax2 = ax1.twinx()
    #ax2.ylim(0, 1.0)
    ax2.plot(x, cumsum, '-ro', alpha=0.5)
    ax2.set_ylabel('', color='r')
    ax2.tick_params('y', colors='r', rotation = 'auto')    
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])    # hide y-labels on right side
    if not show_pct_y:
        ax2.set_yticks([])    
        formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')    
        if title:
            plt.title(title)   
    plt.tight_layout()
    plt.show();