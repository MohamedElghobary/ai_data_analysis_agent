import pandas as pd
import re
import logging
from typing import Dict, Any, Optional
import openai
from config import Config
from src.data_processor import DataProcessor

class QueryProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.data_processor = DataProcessor(dataframe)
        self.setup_openai()
        
    def setup_openai(self):
        """Setup OpenAI client"""
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
        else:
            logging.warning("OpenAI API key not found. AI features will be limited.")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and return results"""
        try:
            # First, try to process with simple pattern matching
            result = self._process_simple_query(query)
            
            if result['success']:
                return result
            
            # If simple processing fails, use AI (if available)
            if Config.OPENAI_API_KEY:
                return self._process_ai_query(query)
            else:
                return {
                    'success': False,
                    'error': 'Complex queries require OpenAI API key. Please configure it in your environment.'
                }
                
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {
                'success': False,
                'error': f'Error processing query: {str(e)}'
            }
    
    def _process_simple_query(self, query: str) -> Dict[str, Any]:
        """Process simple queries using pattern matching"""
        query_lower = query.lower().strip()
        
        # Show top N rows
        if 'top' in query_lower and any(word in query_lower for word in ['rows', 'records', 'entries']):
            n = self._extract_number(query, default=10)
            return {
                'success': True,
                'data': self.df.head(n),
                'explanation': f'Showing top {n} rows from the dataset',
                'code': f'df.head({n})'
            }
        
        # Show basic info
        if any(word in query_lower for word in ['info', 'information', 'overview', 'summary']):
            return {
                'success': True,
                'data': self.data_processor.get_column_info(),
                'explanation': 'Dataset overview and column information',
                'code': 'df.info()'
            }
        
        # Statistical summary
        if any(word in query_lower for word in ['statistics', 'stats', 'describe', 'summary']):
            return {
                'success': True,
                'data': self.df.describe(),
                'explanation': 'Statistical summary of numeric columns',
                'code': 'df.describe()'
            }
        
        # Count rows
        if any(word in query_lower for word in ['count', 'number of rows', 'how many rows']):
            count = len(self.df)
            return {
                'success': True,
                'data': f'Total rows: {count:,}',
                'explanation': f'The dataset contains {count:,} rows',
                'code': 'len(df)'
            }
        
        # Show columns
        if any(word in query_lower for word in ['columns', 'column names', 'fields']):
            return {
                'success': True,
                'data': list(self.df.columns),
                'explanation': 'List of all columns in the dataset',
                'code': 'df.columns.tolist()'
            }
        
        # Missing values
        if any(word in query_lower for word in ['missing', 'null', 'nan', 'empty']):
            missing_data = self.data_processor.analyze_missing_data()
            return {
                'success': True,
                'data': missing_data if not missing_data.empty else 'No missing values found!',
                'explanation': 'Analysis of missing values in the dataset',
                'code': 'df.isnull().sum()'
            }
        
        # Correlation analysis
        if 'correlation' in query_lower or 'corr' in query_lower:
            corr_data = self.data_processor.correlation_analysis()
            return {
                'success': True,
                'data': corr_data,
                'explanation': 'Correlation matrix for numeric columns',
                'code': 'df.corr()'
            }
        
        return {'success': False, 'error': 'Query not recognized'}
    
    def _process_ai_query(self, query: str) -> Dict[str, Any]:
        """Process complex queries using OpenAI"""
        try:
            # Create context about the dataset
            context = self._create_dataset_context()
            
            # Create the prompt
            prompt = self._create_analysis_prompt(query, context)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Generate Python pandas code to answer user queries about their dataset."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            
            # Extract and execute the code
            code = self._extract_code_from_response(response.choices[0].message.content)
            
            if code:
                result = self._execute_pandas_code(code)
                return {
                    'success': True,
                    'data': result,
                    'explanation': response.choices[0].message.content,
                    'code': code
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not generate valid code from the query'
                }
                
        except Exception as e:
            logging.error(f"AI query processing error: {str(e)}")
            return {
                'success': False,
                'error': f'AI processing error: {str(e)}'
            }
    
    def _create_dataset_context(self) -> str:
        """Create context information about the dataset"""
        context = f"""
        Dataset Information:
        - Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns
        - Columns: {', '.join(self.df.columns.tolist())}
        - Data types: {dict(self.df.dtypes)}
        - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
        """
        
        # Add sample data
        context += f"\nFirst 3 rows:\n{self.df.head(3).to_string()}"
        
        return context
    
    def _create_analysis_prompt(self, query: str, context: str) -> str:
        """Create a prompt for AI analysis"""
        return f"""
        {context}
        
        User Query: {query}
        
        Please provide Python pandas code to answer this query. The dataframe is available as 'df'.
        Only return the executable code, wrapped in ```python code blocks.
        Make sure the code is safe and only uses pandas operations.
        """
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from AI response"""
        code_pattern = r'```python\n(.*?)\n```'
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for any code-like content
        lines = response.split('\n')
        code_lines = [line for line in lines if 'df.' in line or any(keyword in line for keyword in ['import', 'print', '='])]
        
        return '\n'.join(code_lines) if code_lines else None
    
    def _execute_pandas_code(self, code: str) -> Any:
        """Safely execute pandas code"""
        # Create a safe execution environment
        safe_dict = {
            'df': self.df,
            'pd': pd,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'round': round
        }
        
        try:
            # Execute the code
            exec(code, safe_dict)
            
            # Try to get the result (look for common result variable names)
            for var_name in ['result', 'output', 'answer']:
                if var_name in safe_dict:
                    return safe_dict[var_name]
            
            # If no specific result variable, return the last evaluated expression
            return eval(code.split('\n')[-1], safe_dict)
            
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
    
    def _extract_number(self, text: str, default: int = 10) -> int:
        """Extract number from text"""
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else default