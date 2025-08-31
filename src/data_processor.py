import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.original_df = dataframe.copy()
        
    def get_column_info(self) -> pd.DataFrame:
        """Get comprehensive information about each column"""
        info_data = []
        
        for column in self.df.columns:
            col_data = self.df[column]
            info = {
                'Column': column,
                'Data Type': str(col_data.dtype),
                'Non-Null Count': col_data.count(),
                'Null Count': col_data.isnull().sum(),
                'Unique Values': col_data.nunique(),
                'Memory Usage (KB)': round(col_data.memory_usage(deep=True) / 1024, 2)
            }
            
            # Add type-specific information
            if pd.api.types.is_numeric_dtype(col_data):
                info.update({
                    'Min': col_data.min(),
                    'Max': col_data.max(),
                    'Mean': round(col_data.mean(), 2) if not col_data.empty else None
                })
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                if col_data.nunique() < 10:  # Show sample values for categorical data
                    info['Sample Values'] = ', '.join(map(str, col_data.dropna().unique()[:5]))
            
            info_data.append(info)
        
        return pd.DataFrame(info_data)
    
    def analyze_missing_data(self) -> pd.DataFrame:
        """Analyze missing data patterns"""
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if missing_data.empty:
            return pd.DataFrame()
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': (missing_data.values / len(self.df) * 100).round(2)
        })
        
        return missing_df.sort_values('Missing Count', ascending=False)
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive data quality report"""
        report = {}
        
        # Basic metrics
        report['Total Records'] = f"{len(self.df):,}"
        report['Total Columns'] = len(self.df.columns)
        report['Duplicate Rows'] = self.df.duplicated().sum()
        report['Complete Records'] = (len(self.df) - self.df.isnull().any(axis=1).sum())
        
        # Data type distribution
        dtype_counts = self.df.dtypes.value_counts()
        report['Numeric Columns'] = dtype_counts.get('int64', 0) + dtype_counts.get('float64', 0)
        report['Text Columns'] = dtype_counts.get('object', 0)
        report['Date Columns'] = dtype_counts.get('datetime64[ns]', 0)
        
        return report
    
    def suggest_analysis(self) -> List[str]:
        """Suggest potential analyses based on data characteristics"""
        suggestions = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 2:
            suggestions.append("Correlation analysis between numeric variables")
            suggestions.append("Statistical summary of numeric columns")
        
        if len(categorical_cols) > 0:
            suggestions.append("Frequency analysis of categorical variables")
        
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            suggestions.append("Group-by analysis (numeric by categories)")
        
        # Check for potential time series
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            suggestions.append("Time series analysis")
        
        # Check for potential outliers
        if len(numeric_cols) > 0:
            suggestions.append("Outlier detection and analysis")
        
        return suggestions
    
    def basic_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate basic statistics for specified columns or all numeric columns"""
        if columns is None:
            return self.df.describe()
        else:
            return self.df[columns].describe()
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Perform correlation analysis on numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def value_counts(self, column: str, top_n: int = 10) -> pd.Series:
        """Get value counts for a specific column"""
        return self.df[column].value_counts().head(top_n)
    
    def filter_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Filter data based on conditions"""
        filtered_df = self.df.copy()
        
        for column, condition in conditions.items():
            if column in filtered_df.columns:
                if isinstance(condition, dict):
                    # Handle range conditions
                    if 'min' in condition:
                        filtered_df = filtered_df[filtered_df[column] >= condition['min']]
                    if 'max' in condition:
                        filtered_df = filtered_df[filtered_df[column] <= condition['max']]
                elif isinstance(condition, list):
                    # Handle multiple values
                    filtered_df = filtered_df[filtered_df[column].isin(condition)]
                else:
                    # Handle single value
                    filtered_df = filtered_df[filtered_df[column] == condition]
        
        return filtered_df
    
    def group_analysis(self, group_by_col: str, agg_col: str, agg_func: str = 'mean') -> pd.DataFrame:
        """Perform group-by analysis"""
        if group_by_col not in self.df.columns or agg_col not in self.df.columns:
            raise ValueError("Specified columns not found in dataframe")
        
        return self.df.groupby(group_by_col)[agg_col].agg(agg_func).reset_index()