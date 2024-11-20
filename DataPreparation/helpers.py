from IPython.core.display import HTML,display
from IPython.display import Markdown


def display_with_scroll(df):
    """
    Display a Pandas DataFrame with horizontal scrolling in Jupyter Notebook.

    Parameters:
        df : pandas.DataFrame
            The DataFrame to display.
    """
    display(HTML(df.to_html(classes='table table-striped', justify='left', border=0)))
    # CSS for scrolling
    display(HTML('''
    <style>
        .dataframe-div {
            overflow-x: auto;
            white-space: nowrap;
            width: 100%;
        }
        .dataframe {
            width: auto;
        }
    </style>
    '''))
    
def convey_insights(bullets_arr):
    '''
    Give it a bullet points array, give you bullet points in markdown for insights.
    '''
    # make a markdown string with the bullets
    markdown_str = '<h3><font color="pink" size=5>Insights</font></h3> <font size=4>\n'
    
    for bullet in bullets_arr:
        markdown_str += '<font color="pink">✦</font> ' + bullet + '<br>' + '<br>'
    # display the markdown string
    markdown_str += '</font>'
    display(Markdown(markdown_str))