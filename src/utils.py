import pandas as pd
import streamlit as st
import os
from typing import Optional, Dict, Any
from config import Config
import tempfile

class FileHandler:
    """Handle file operations for data uploads"""
    
    @staticmethod
    def process_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Process uploaded file and return DataFrame"""
        try:
            # Get file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Read file based on extension
            if file_extension == '.csv':
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        return df
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not read CSV file with any supported encoding")
                
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
                return df
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return None

class SessionManager:
    """Manage Streamlit session state"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'file_name' not in st.session_state:
            st.session_state.file_name = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        # Ensure data is properly initialized
        if hasattr(st.session_state, 'data') and st.session_state.data is None:
            st.session_state.data = None
    
    @staticmethod
    def add_to_history(query: str, result: Dict[str, Any]):
        """Add query and result to history"""
        st.session_state.query_history.append({
            'query': query,
            'result': result,
            'timestamp': pd.Timestamp.now()
        })
    
    @staticmethod
    def clear_session():
        """Clear all session data"""
        for key in st.session_state.keys():
            del st.session_state[key]
        SessionManager.initialize_session()

class DataValidator:
    """Validate data quality and structure"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame and return validation report"""
        validation_report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_report['errors'].append("DataFrame is empty")
            validation_report['is_valid'] = False
            return validation_report
        
        # Check for extremely large datasets
        if len(df) > 1000000:
            validation_report['warnings'].append(
                f"Large dataset ({len(df):,} rows). Consider sampling for better performance."
            )
        
        # Check for high memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_usage_mb > 500:
            validation_report['warnings'].append(
                f"High memory usage ({memory_usage_mb:.1f} MB). Consider data optimization."
            )
        
        # Check for missing values
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 50:
            validation_report['warnings'].append(
                f"High percentage of missing values ({missing_percentage:.1f}%)"
            )
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_report['warnings'].append(
                f"Found {duplicate_count} duplicate rows"
            )
        
        # Check for columns with all missing values
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            validation_report['warnings'].append(
                f"Columns with all missing values: {', '.join(all_null_cols)}"
            )
        
        # Recommendations
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            validation_report['recommendations'].append("Consider correlation analysis between numeric variables")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            validation_report['recommendations'].append("Explore categorical variable distributions")
        
        return validation_report
