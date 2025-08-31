import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_processor import DataProcessor
from src.query_processor import QueryProcessor
from src.visualizer import DataVisualizer
from src.utils import FileHandler, SessionManager
from config import Config
import os

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
Config.create_directories()

# Initialize session state
SessionManager.initialize_session()

def main():
    st.title("🤖 AI Data Analysis Agent")
    st.markdown("**Analyze your data using natural language queries - No SQL expertise required!**")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files (Max 200MB)"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            with st.spinner("Processing file..."):
                file_handler = FileHandler()
                df = file_handler.process_uploaded_file(uploaded_file)
                
                if df is not None:
                    st.session_state.data = df
                    st.session_state.file_name = uploaded_file.name
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    
                    # Display basic info
                    st.subheader("📋 Data Overview")
                    st.write(f"**Rows:** {len(df):,}")
                    st.write(f"**Columns:** {len(df.columns)}")
                    st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.1f} MB")
                else:
                    st.error("❌ Failed to process the file. Please check the file format and try again.")
                    if 'data' in st.session_state:
                        st.session_state.data = None
        
        # Settings section
        st.header("⚙️ Settings")
        show_sample = st.checkbox("Show data sample", value=True)
        max_rows = st.slider("Max rows to display", 100, 5000, 1000)
    
    # Main content area
    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Query & Analysis", "📈 Visualizations", "📋 Data Summary"])
        
        with tab1:
            show_data_overview(df, show_sample, max_rows)
        
        with tab2:
            show_query_interface(df)
        
        with tab3:
            show_visualizations(df)
        
        with tab4:
            show_data_summary(df)
    else:
        # Welcome screen
        show_welcome_screen()

def show_data_overview(df, show_sample, max_rows):
    st.subheader("📊 Data Overview")
    
    # Data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Column information
    st.subheader("📋 Column Information")
    data_processor = DataProcessor(df)
    column_info = data_processor.get_column_info()
    st.dataframe(column_info, use_container_width=True)
    
    # Data sample
    if show_sample:
        st.subheader("🔍 Data Sample")
        display_df = df.head(min(max_rows, len(df)))
        st.dataframe(display_df, use_container_width=True)

def show_query_interface(df):
    st.subheader("🔍 Natural Language Query Interface")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - "Show me the top 10 rows"
        - "What is the average of column X?"
        - "Show me data where column Y is greater than 100"
        - "Create a summary statistics for all numeric columns"
        - "Filter data where column Z contains 'keyword'"
        - "Show me the correlation between columns A and B"
        """)
    
    # Query input
    query = st.text_area(
        "Ask a question about your data:",
        placeholder="e.g., Show me the top 10 customers by sales amount",
        height=100
    )
    
    if st.button("🚀 Analyze", type="primary"):
        if query.strip():
            with st.spinner("Processing your query..."):
                query_processor = QueryProcessor(df)
                result = query_processor.process_query(query)
                
                if result['success']:
                    st.success("✅ Query processed successfully!")
                    
                    # Display explanation
                    if result.get('explanation'):
                        st.info(f"**Analysis:** {result['explanation']}")
                    
                    # Display result
                    if result.get('data') is not None:
                        if isinstance(result['data'], pd.DataFrame):
                            st.dataframe(result['data'], use_container_width=True)
                        else:
                            st.write(result['data'])
                    
                    # Display code (optional)
                    if result.get('code') and st.checkbox("Show generated code"):
                        st.code(result['code'], language='python')
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
        else:
            st.warning("Please enter a query!")

def show_visualizations(df):
    st.subheader("📈 Data Visualizations")
    
    visualizer = DataVisualizer(df)
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["Auto Suggest", "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
    )
    
    if viz_type == "Auto Suggest":
        st.info("💡 Based on your data, here are some recommended visualizations:")
        suggestions = visualizer.suggest_visualizations()
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    
    elif viz_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("X-axis:", df.columns)
        with col2:
            y_column = st.selectbox("Y-axis:", df.select_dtypes(include=['number']).columns)
        
        if st.button(f"Create {viz_type}"):
            fig = visualizer.create_chart(viz_type.lower().replace(' ', '_'), x_column, y_column)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        column = st.selectbox("Select column:", df.select_dtypes(include=['number']).columns)
        if st.button("Create Histogram"):
            fig = visualizer.create_histogram(column)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        column = st.selectbox("Select column:", df.select_dtypes(include=['number']).columns)
        if st.button("Create Box Plot"):
            fig = visualizer.create_boxplot(column)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        if st.button("Create Correlation Heatmap"):
            fig = visualizer.create_correlation_heatmap()
            st.plotly_chart(fig, use_container_width=True)

def show_data_summary(df):
    st.subheader("📋 Data Summary & Statistics")
    
    data_processor = DataProcessor(df)
    
    # Basic statistics
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values analysis
    st.subheader("🔍 Missing Values Analysis")
    missing_data = data_processor.analyze_missing_data()
    if not missing_data.empty:
        st.dataframe(missing_data, use_container_width=True)
    else:
        st.success("✅ No missing values found!")
    
    # Data quality report
    st.subheader("✅ Data Quality Report")
    quality_report = data_processor.get_data_quality_report()
    for metric, value in quality_report.items():
        st.metric(metric, value)

def show_welcome_screen():
    st.markdown("""
    ## Welcome to AI Data Analysis Agent! 🚀
    
    This intelligent agent helps you analyze your data using natural language queries. 
    No SQL knowledge required!
    
    ### 🌟 Key Features:
    - **📁 File Upload**: Support for CSV and Excel files
    - **🤖 Natural Language Queries**: Ask questions in plain English
    - **📊 Automatic Visualizations**: Generate charts and graphs
    - **📋 Statistical Analysis**: Get comprehensive data insights
    - **🔍 Data Quality Assessment**: Identify issues and patterns
    
    ### 🚀 Getting Started:
    1. **Upload your data** using the sidebar file uploader
    2. **Explore your data** in the Data Overview tab
    3. **Ask questions** using natural language in the Query tab
    4. **Create visualizations** in the Visualizations tab
    5. **Review statistics** in the Data Summary tab
    
    ### 💡 Example Questions You Can Ask:
    - "What are the top 10 customers by revenue?"
    - "Show me the correlation between price and sales"
    - "Filter data where status is 'completed'"
    - "What's the average order value by region?"
    - "Create a chart showing monthly trends"
    
    **Upload a file to get started!** 👆
    """)

if __name__ == "__main__":
    main()