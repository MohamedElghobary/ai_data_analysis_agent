import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np

class DataVisualizer:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
    def suggest_visualizations(self) -> List[str]:
        """Suggest appropriate visualizations based on data characteristics"""
        suggestions = []
        
        if len(self.numeric_columns) >= 2:
            suggestions.append("📊 Correlation Heatmap - Shows relationships between numeric variables")
            suggestions.append("📈 Scatter Plot - Compare two numeric variables")
        
        if len(self.numeric_columns) >= 1:
            suggestions.append("📊 Histogram - Distribution of numeric variables")
            suggestions.append("📦 Box Plot - Identify outliers and distribution shape")
        
        if len(self.categorical_columns) >= 1:
            suggestions.append("📊 Bar Chart - Frequency of categorical variables")
        
        if len(self.categorical_columns) >= 1 and len(self.numeric_columns) >= 1:
            suggestions.append("📊 Grouped Bar Chart - Numeric values by categories")
        
        # Check for time-based data
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            suggestions.append("📈 Time Series Plot - Trends over time")
        
        return suggestions
    
    def create_chart(self, chart_type: str, x_column: str, y_column: str, **kwargs) -> go.Figure:
        """Create different types of charts"""
        
        if chart_type == 'bar_chart':
            return self.create_bar_chart(x_column, y_column, **kwargs)
        elif chart_type == 'line_chart':
            return self.create_line_chart(x_column, y_column, **kwargs)
        elif chart_type == 'scatter_plot':
            return self.create_scatter_plot(x_column, y_column, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def create_bar_chart(self, x_column: str, y_column: str, title: Optional[str] = None) -> go.Figure:
        """Create a bar chart"""
        fig = px.bar(
            self.df, 
            x=x_column, 
            y=y_column,
            title=title or f"{y_column} by {x_column}",
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            showlegend=False
        )
        return fig
    
    def create_line_chart(self, x_column: str, y_column: str, title: Optional[str] = None) -> go.Figure:
        """Create a line chart"""
        fig = px.line(
            self.df, 
            x=x_column, 
            y=y_column,
            title=title or f"{y_column} over {x_column}",
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title()
        )
        return fig
    
    def create_scatter_plot(self, x_column: str, y_column: str, color_column: Optional[str] = None, 
                           size_column: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create a scatter plot"""
        fig = px.scatter(
            self.df, 
            x=x_column, 
            y=y_column,
            color=color_column,
            size=size_column,
            title=title or f"{y_column} vs {x_column}",
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title()
        )
        return fig
    
    def create_histogram(self, column: str, bins: int = 30, title: Optional[str] = None) -> go.Figure:
        """Create a histogram"""
        fig = px.histogram(
            self.df, 
            x=column,
            nbins=bins,
            title=title or f"Distribution of {column}",
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title=column.replace('_', ' ').title(),
            yaxis_title='Frequency',
            showlegend=False
        )
        return fig
    
    def create_boxplot(self, column: str, group_by: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create a box plot"""
        fig = px.box(
            self.df, 
            y=column,
            x=group_by,
            title=title or f"Box Plot of {column}",
            template='plotly_white'
        )
        fig.update_layout(
            yaxis_title=column.replace('_', ' ').title(),
            xaxis_title=group_by.replace('_', ' ').title() if group_by else ''
        )
        return fig
    
    def create_correlation_heatmap(self, title: Optional[str] = None) -> go.Figure:
        """Create a correlation heatmap"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
        
        corr_matrix = self.df[self.numeric_columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title=title or "Correlation Heatmap",
            template='plotly_white',
            color_continuous_scale='RdBu_r'
        )
        return fig
    
    def create_pie_chart(self, column: str, max_categories: int = 10, title: Optional[str] = None) -> go.Figure:
        """Create a pie chart for categorical data"""
        value_counts = self.df[column].value_counts().head(max_categories)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title or f"Distribution of {column}",
            template='plotly_white'
        )
        return fig
    
    def create_time_series(self, date_column: str, value_column: str, title: Optional[str] = None) -> go.Figure:
        """Create a time series plot"""
        fig = px.line(
            self.df, 
            x=date_column, 
            y=value_column,
            title=title or f"{value_column} over Time",
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_column.replace('_', ' ').title()
        )
        return fig
    
    def create_grouped_bar_chart(self, x_column: str, y_column: str, group_column: str, 
                                title: Optional[str] = None) -> go.Figure:
        """Create a grouped bar chart"""
        fig = px.bar(
            self.df, 
            x=x_column, 
            y=y_column,
            color=group_column,
            title=title or f"{y_column} by {x_column} and {group_column}",
            template='plotly_white',
            barmode='group'
        )
        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title()
        )
        return fig
    
    def create_violin_plot(self, column: str, group_by: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create a violin plot"""
        fig = px.violin(
            self.df, 
            y=column,
            x=group_by,
            title=title or f"Violin Plot of {column}",
            template='plotly_white'
        )
        fig.update_layout(
            yaxis_title=column.replace('_', ' ').title(),
            xaxis_title=group_by.replace('_', ' ').title() if group_by else ''
        )
        return fig
