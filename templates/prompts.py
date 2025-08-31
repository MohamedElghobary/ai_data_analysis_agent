class PromptTemplates:
    """Templates for AI prompts"""
    
    SYSTEM_PROMPT = """
    You are an expert data analyst. Your task is to analyze datasets and answer user questions about their data.
    You should provide clear, actionable insights and generate appropriate Python pandas code when needed.
    Always prioritize accuracy and explain your reasoning.
    """
    
    ANALYSIS_PROMPT = """
    Dataset Context:
    {context}
    
    User Question: {query}
    
    Please analyze the question and provide:
    1. A clear explanation of what analysis is needed
    2. Python pandas code to perform the analysis (if applicable)
    3. Interpretation of the results
    
    Format your response clearly and include any relevant insights or recommendations.
    """
    
    VISUALIZATION_PROMPT = """
    Dataset Context:
    {context}
    
    User Request: {query}
    
    Please suggest the most appropriate visualization(s) for this request and provide:
    1. Explanation of why this visualization is suitable
    2. Python code using plotly or matplotlib
    3. Any customization recommendations
    """
    
    CODE_GENERATION_PROMPT = """
    You are a Python pandas expert. Generate clean, efficient pandas code for the following request:
    
    Dataset Info:
    - Columns: {columns}
    - Shape: {shape}
    - Data Types: {dtypes}
    
    Request: {query}
    
    Requirements:
    1. Use only pandas operations
    2. Handle potential errors gracefully
    3. Return only executable code
    4. Add comments for complex operations
    """
