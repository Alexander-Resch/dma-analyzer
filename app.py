import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
from scipy.signal import detrend, savgol_filter
import csv as cf
from io import StringIO

def main():

    header = 0
    uploaded_file = None

    def find_header():
        with StringIO(uploaded_file.getvalue().decode("utf-8")) as fin:
            reader = cf.reader(fin)
            idx = next(idx for idx, row in enumerate(reader) if len(row) > 1)  # 4
        return idx

    @st.cache(persist=True)
    def load_data(header):
        data = pd.read_csv(uploaded_file,header = find_header()+header)
        return data


    @st.cache(persist=True)
    def reduce_data(data,header,disp_col,load_col):
        data = data.iloc[header:,[disp_col,load_col]]
        return data


    @st.cache(persist=True)
    def plot_time_series_plotly(df):
        fig = px.line(df,title='Load and Displacement Data')#, x = df.columns[0], y = df.columns[1])
        return fig

    @st.cache(persist=True)
    def plot_ellipse_plotly(df):

        if 'Filtered' in df.columns:
            y = 'Filtered'
        else:
            y = df.columns[1]

        fig = px.scatter(df,
                         x=df.columns[0],
                         y=y,
                         title='Load and Displacement Data',
                         opacity=0.3
                         )
        return fig

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("DMA Data Analyzer")

    st.sidebar.title("DMA Data Analyzer")

    uploaded_file = st.sidebar.file_uploader("Drag and Drop or Select Data")

    if uploaded_file is None:
        st.header('??? Please upload a .csv file.')
        st.sidebar.header('??? Please upload a .csv file.')
        st.stop()

    df = load_data(header)

    header = int(st.sidebar.number_input('header size - adjust manually if needed', step=1, key='header_size'))
    df = load_data(header)

    columns = st.sidebar.multiselect("Select Columns - Force and Displacement",
                                     options=tuple([i for i in df.columns]),
                                     default=tuple([i for i in df.columns if i in df.columns[3:5]])
                                     )

    drop_first = int(st.sidebar.number_input('Drop first n rows', min_value=100, step=100, key='drop_first'))
    keep = int(st.sidebar.number_input('Keep m rows', value=len(df), min_value=0, max_value=50000, step=100, key='keep'))

    df_reduced = df[columns].iloc[drop_first:drop_first+keep].astype(float)





    st.sidebar.subheader("Data Processing")
    if st.sidebar.checkbox('Detrend', False):
        #df_reduced[df_reduced.columns[1]] = detrend(df_reduced[df_reduced.columns[1]])

        df_reduced['Filtered'] = detrend(df_reduced[df_reduced.columns[1]])

    if st.sidebar.checkbox('Savitzky-Golay', False):
        order = st.sidebar.slider('Savitzky Golay Polynomial Order', value=1, min_value=1,step=2,max_value=31)


        window = st.sidebar.slider('Savitzky Golay Window Length', value=3, min_value=order+2,step=1,max_value=33)

        if 'Filtered' in df_reduced.columns:

            df_reduced['Filtered'] = savgol_filter(df_reduced['Filtered'],
                                                          polyorder=order,
                                                          window_length=window)
        else:
            df_reduced['Filtered'] = savgol_filter(df_reduced[df_reduced.columns[1]],
                                                          polyorder=order,
                                                          window_length=window)

    if st.sidebar.checkbox("Show Data", True):
        st.subheader(uploaded_file.name + ' - Preview')
        st.write(df.head())
        st.subheader('Selected Data - Preview')

        st.write(df_reduced.head())

    f1 = plot_time_series_plotly(df_reduced)
    st.plotly_chart(f1, use_container_width=True)

    f2 = plot_ellipse_plotly(df_reduced)
    st.plotly_chart(f2, use_container_width=True)


    if st.sidebar.button("Classify", key='classify'):
        st.write("That's it, folks!")


if __name__ == '__main__':

    main()