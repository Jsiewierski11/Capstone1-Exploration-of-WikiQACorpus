import pandas as pd 
import numpy as np 


def select_doc_titles(df, titles):
    '''
    Parameters:
        df - pandas dataframe from which we are selecting the columns from
        titles - an np array of document titles to select
    Returns:
        Pandas Dataframe of only desired document titles
    '''
    series = np.array.shape(titles.shape)
    for i in range(0, len(titles)):
        series.append(df[df['DocumentTitle'] == titiles[i]].reset_index())

    