import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from scipy import stats
from scipy.stats import skew
from autoviz.AutoViz_Class import AutoViz_Class
import matplotlib.pyplot as plt
import os
#Set background color
st.markdown(
     """
     <style>
     .main{
         background-color:#ADD8E6;
     }
     </style>
     """,
     unsafe_allow_html=True
 )

st.title('Detailed Exploratory Data Analysis App')

st.markdown(""" 
This app performs detailed Exploratory Data Analysis (EDA).
**Python libraries:** streamlit, pandas, plotly, numpy, scipy
**Need to contact:** [SeaportAI.com](sananancy24@gmail.com). 
""")

#Function to save dataframe to csv file
def save_dataframe(data,filename="modified_dataset.csv"):
    data.to_csv(filename,index=False)
    return filename


file_bytes = st.file_uploader("Upload a CSV file", type="csv")

if file_bytes is not None:
    data = pd.read_csv(file_bytes)

    #Add CSS to ensure the button is at the bottom of the sidebar
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content{
            display: flex;
            flex-direction:column;
            justify-content:flex-end;
        }
        </style>
        """,
        unsafe_allow_html=True
     )


    if st.sidebar.button('Generate EDA Report'):
        
        with open("temp_file_bytes.csv","wb") as f:
            f.write(file_bytes.getbuffer())

        st.subheader('EDA Report')
        AV= AutoViz_Class()
        EDA_report=AV.AutoViz(filename="temp_file_bytes.csv",sep=',',depVar='',dfte=data,header=0,verbose=0,
                              lowess=False,chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=30)
       
        figs=[plt.figure(n) for n in plt.get_fignums()]
        for i ,fig in enumerate(figs):
            plot_path=f"autoviz_plot_{i}.svg"
            fig.savefig(plot_path)
            st.image(plot_path)

        for i in range(len(figs)):
            os.remove(f"autoviz_plot_{i}.svg")

        os.remove("temp_file_bytes.csv")

        
    # Display the dataset
    st.subheader('Uploaded Dataset')
    st.dataframe(data.head(11))
    
    # Identify column types
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    #show basic information
    st.subheader('Basic Data Quality Steps')

    if st.checkbox('Show shape'):
        st.write(data.shape)

    if st.checkbox('Show columns'):
        st.write(data.columns.tolist())

    if st.checkbox('Show summary'):
        st.write(data.describe())

    
    if st.checkbox('Show missing values'):
        st.write(data.isnull().sum())
    
   
    if st.checkbox('Show duplicate rows'):
        st.write('Number of duplicate rows:')
        st.write(data.duplicated().sum())

    if st.checkbox('Show data types'):
        st.write(data.dtypes)

    #Outlier detection
    st.subheader('Outlier Detection')

    if st.checkbox('Show boxplots for outlier detection'):
        numeric_columns=data.select_dtypes(include=['float64','int64']).columns
        selected_column=st.selectbox('Select column for boxplot',numeric_columns)

        if selected_column:
            fig=px.box(data,y=selected_column,points="all")
            st.plotly_chart(fig)
   
    if st.checkbox('Show scatter plots for outlier detection'):
        numeric_columns=data.select_dtypes(include=['float64','int64']).columns
        x_column= st.selectbox('Select X-axis column for scatter plot',numeric_columns)
        y_column=st.selectbox('Select Y-axis column for scatter plot',numeric_columns)

        if x_column and y_column:
            fig=px.scatter(data,x=x_column, y=y_column)
            st.plotly_chart(fig)

    
    # Skewness Check
    st.subheader('Skewness Check')
    if st.checkbox('Show skewness of numeric columns'):
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        skewness_values = data[numeric_columns].apply(lambda x: skew(x.dropna()))
        st.write(skewness_values)
        fig = px.bar(skewness_values, title='Skewness of Numeric Columns')
        st.plotly_chart(fig)

    # Check for Spelling Errors in Column Names
    st.subheader('Check for Spelling Errors in Column Names')
    if st.checkbox('Check for unusual characters in column names'):
        unusual_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', '=', '/', '\\', '|', '{', '}', '[', ']', ':', ';', '"', "'", '<', '>', ',', '?', '`', '~']
        columns_with_unusual_chars = [col for col in data.columns if any(char in col for char in unusual_chars)]
        if columns_with_unusual_chars:
            st.write('Columns with unusual characters:', columns_with_unusual_chars)
        else:
            st.write('No unusual characters found in column names.')

    # Check for Negative Values
    st.subheader('Check for Negative Values in Numeric Columns')
    if st.checkbox('Show negative values in numeric columns'):
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        negative_values = data[numeric_columns].apply(lambda x: (x < 0).sum())
        st.write(negative_values)
        st.write('Columns with negative values:')
        st.write(negative_values[negative_values > 0])

    # Fill Missing Values and save filled dataset
    st.subheader('Fill Missing Values')
    if st.checkbox('Fill missing values in a column'):
       fill_column = st.selectbox('Select column to fill missing values', data.columns)
       fill_method = st.selectbox('Select fill method', ['Mean', 'Mode'])

       if fill_method == 'Mean' and data[fill_column].dtype in ['int64', 'float64']:
          data[fill_column].fillna(data[fill_column].mean(), inplace=True)
          st.write(f'Filled missing values in {fill_column} using {fill_method} method.')
       elif fill_method == 'Mode' and data[fill_column].dtype == 'object':
           data[fill_column].fillna(data[fill_column].mode()[0], inplace=True)
           st.write(f'Filled missing values in {fill_column} using {fill_method} method.')
       else:
           st.write("Selected method is not applicable for the chosen column type.")

       st.write(data[fill_column].isnull().sum())  # Verify if missing values are filled
       st.write(data.head())  # Display updated dataframe

       
       # Save modified dataset
       if st.checkbox('Save filled dataset'):
          filename=save_dataframe(data)
          st.write(f'saved filled dataset as {filename}')

          #Load the save dataset
          saved_data=pd.read_csv(filename)
           
    # Function to generate histogram
    def generate_histogram(data, column):
        fig = px.histogram(data, x=column, marginal="box", nbins=30)
        return fig

    # Function to generate boxplot
    def generate_boxplot(data, column):
        fig = px.box(data, y=column)
        return fig
    
    # Function to generate bar chart
    def generate_bar_chart(data, column):
       fig = px.bar(data, x=column)
       return fig

    # Function to generate pie chart
    def generate_pie_chart(data, column):
        if saved_data[column].nunique() <= 10:  # Ensuring the column has a reasonable number of unique values
           fig = px.pie(data, names=column)
           return fig
        else:
            return None

        
    
    # Sidebar for graphical and statistical analysis
    analysis_type = st.sidebar.selectbox('Select Analysis Type', ['Graphical Analysis', 'Statistical Analysis'])

    if analysis_type == 'Graphical Analysis':
        st.header('Graphical Analysis')
        # Univariate Analysis
        st.sidebar.subheader('Univariate Analysis')
        column = st.sidebar.selectbox('Select a column for univariate analysis', data.columns)

        if column:
           st.subheader(f'Univariate Analysis for {column}')
           plot_type = st.sidebar.radio('Select plot type', ('Histogram', 'Boxplot', 'Bar Chart', 'Pie Chart'))

           if plot_type == 'Histogram':
              fig = generate_histogram(data, column)
              st.plotly_chart(fig)

           elif plot_type == 'Boxplot':
               fig = generate_boxplot(data, column)
               st.plotly_chart(fig)

           elif plot_type == 'Bar Chart':
               fig = generate_bar_chart(data, column)
               st.plotly_chart(fig)

           elif plot_type == 'Pie Chart':
               fig = generate_pie_chart(data, column)
               if fig:
                   st.plotly_chart(fig)
               else:
                   st.write("Too many unique values for a pie chart")

        # Function to generate stack bar chart
        def generate_stack_bar_chart(data, x_column, y_column):
            fig = px.bar(data, x=x_column, y=y_column, color=y_column, barmode='stack')
            return fig

        # Function to generate line chart
        def generate_line_chart(data, x_column, y_column):
            fig = px.line(data, x=x_column, y=y_column)
            return fig

        # Function to generate scatter plot
        def generate_scatter_plot(data, x_column, y_column):
            fig = px.scatter(data, x=x_column, y=y_column)
            return fig

        # Bivariate Analysis
        st.sidebar.subheader('Bivariate Analysis')
        x_column = st.sidebar.selectbox('Select X-axis column', data.columns)
        y_column = st.sidebar.selectbox('Select Y-axis column', data.columns)

        if x_column != y_column:
           st.subheader(f'Bivariate Analysis for {x_column} and {y_column}')
           plot_type = st.sidebar.radio('Select plot type', ('Stacked Bar Chart', 'Line Chart', 'Scatter Plot'))

           if plot_type == 'Stacked Bar Chart':
              fig = generate_stack_bar_chart(data, x_column, y_column)
              st.plotly_chart(fig)

           elif plot_type == 'Line Chart':
              fig = generate_line_chart(data, x_column, y_column)
              st.plotly_chart(fig)

           elif plot_type == 'Scatter Plot':
              fig = generate_scatter_plot(data, x_column, y_column)
              st.plotly_chart(fig)

        # Function to generate correlation matrix
        def generate_correlation_matrix(df):
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            correlation_matrix = numeric_df.corr()
            fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
            return fig

        # Function to generate heatmap
        def generate_heatmap(df):
            fig = px.imshow(df, text_auto=True, aspect="auto")
            return fig

        # Multivariate Analysis
        st.sidebar.subheader('Multivariate Analysis')
        multivariate_plot_type = st.sidebar.radio('Select plot type', ('Correlation Matrix', 'Heatmap'))

        if multivariate_plot_type == 'Correlation Matrix':
           st.subheader('Correlation Matrix')
           fig = generate_correlation_matrix(data)
           st.plotly_chart(fig)

        elif multivariate_plot_type == 'Heatmap':
           st.subheader('Heatmap')
           fig = generate_heatmap(data)
           st.plotly_chart(fig)

            
    elif analysis_type == 'Statistical Analysis':
        st.header('Statistical Analysis')
        
        
        # Descriptive statistics
        st.subheader('Descriptive Statistics')
        st.write(data.describe(include='all'))
        
        # Inferential statistics
        st.subheader('Inferential Statistics')

                
        #T-test for numerical variables
        st.subheader('T-test')


        if len(numerical_cols) >= 2:
            num_col1 = st.selectbox('Select first numerical variable for t-test', numerical_cols, key='ttest1')
            num_col2 = st.selectbox('Select second numerical variable for t-test', numerical_cols, key='ttest2')
            if num_col1 != num_col2:
                t_stat, p_val = stats.ttest_ind(data[num_col1].dropna(), data[num_col2].dropna())
                st.write(f"T-test between {num_col1} and {num_col2}:")
                st.write(f"T-statistic: {t_stat}, P-value: {p_val}")
            else:
                st.write("Please select two different numerical variables for the t-test.")

        #F test for numerical variables
        st.subheader('F-test')


        if len(numerical_cols) >= 2:
           num_col1 = st.selectbox('Select first numerical variable for F-test', numerical_cols, key='ftest1')
           num_col2 = st.selectbox('Select second numerical variable for F-test', numerical_cols, key='ftest2')
           if num_col1 != num_col2:
              var1 = data[num_col1].dropna().var()
              var2 = data[num_col2].dropna().var()
              f_stat = var1 / var2
              dfn = len(data[num_col1].dropna()) - 1
              dfd = len(data[num_col2].dropna()) - 1
              p_val = 1 - stats.f.cdf(f_stat, dfn, dfd)
        
              st.write(f"F-test between {num_col1} and {num_col2}:")
              st.write(f"F-statistic: {f_stat}, P-value: {p_val}")
           else:
               st.write("Please select two different numerical variables for the F-test.")

        
        # Chi-square test for categorical variables
        st.subheader('Chi-squared test')


        if len(categorical_cols) >= 2:
            cat_col1 = st.selectbox('Select first categorical variable for Chi-square test', categorical_cols, key='chi2_1')
            cat_col2 = st.selectbox('Select second categorical variable for Chi-square test', categorical_cols, key='chi2_2')
            if cat_col1 != cat_col2:
                contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f"Chi-square test between {cat_col1} and {cat_col2}:")
                st.write(f"Chi2 statistic: {chi2}, P-value: {p}, Degrees of freedom: {dof}")
                st.write("Expected frequencies:")
                st.write(expected)
            else:
                st.write("Please select two different categorical variables for the Chi-square test.")

        #Spiro wilk test for normality check
        st.subheader('Shapiro-Wilk test')



        if len(numerical_cols) >= 1:
           num_col = st.selectbox('Select a numerical variable for Shapiro-Wilk test', numerical_cols, key='shapirowilk')
           shapiro_stat, p_val = stats.shapiro(data[num_col].dropna())
    
           st.write(f"Shapiro-Wilk test for {num_col}:")
           st.write(f"Shapiro-Wilk statistic: {shapiro_stat}, P-value: {p_val}")

#if st.button("Generate EDA Report"):
    #figs=generate_report(data)
    #st.markdown(download_report(figs,stats_summary=data.describe()),unsafe_allow_html=True)

