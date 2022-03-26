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

def main():

    header = 0
    @st.cache(persist=True)
    def load_data(header):
        data = pd.read_csv(uploaded_file,header = header)
        return data

    @st.cache(persist=True)
    def reduce_data(data,header,disp_col,load_col):
        data = data.iloc[header:,[disp_col,load_col]]
        return data

    def plot_time_series(df):
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        return f, ax

    #def plot_time_series_plotly(df):


    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("DMA Data Analyzer")

    st.sidebar.title("DMA Data Analyzer")

    uploaded_file = st.sidebar.file_uploader("Drag and Drop or Select Data")
    df = load_data(header)

    header = int(st.sidebar.number_input('header size', step=1, key='header_size'))
    df = load_data(header)

    columns = st.sidebar.multiselect("Select Columns",
                                     tuple([i for i in df.columns])
                                     )

    col1, col2 = st.columns(2)
    drop_first = int(st.sidebar.number_input('Drop first n rows', min_value=100, step=100, key='drop_first'))
    drop_last = int(st.sidebar.number_input('Drop last m rows', min_value=0, step=100, key='drop_last'))

    df_reduced = df[columns].iloc[drop_first:len(df)-drop_last].astype(float)

    if st.sidebar.checkbox("Show Data", True):
        st.subheader(uploaded_file.name + ' - Preview')
        st.write(df.head())
        st.subheader('Selected Data - Preview')

        st.write(df_reduced.head())

    f1, ax1 = plot_time_series(df_reduced)
    df_reduced.plot(ax=ax1)
    st.pyplot(f1)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         (
                                          'Confusion Matrix',
                                          'ROC Curve',
                                          'Precision-Recall Curve'
                                          )
                                         )
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)




if __name__ == '__main__':

    main()