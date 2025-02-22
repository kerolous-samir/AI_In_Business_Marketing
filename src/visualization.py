import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def barplot_visualization(df, column):
    fig = px.bar(x=df[column].value_counts().index, y=df[column].value_counts(), 
                 color=df[column].value_counts().index, height=df[column].value_counts().max())
    fig.show()

def plot_heatmap(df):
    corr_matrix = df.iloc[:, :10].corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, cbar=False)
    plt.show()
