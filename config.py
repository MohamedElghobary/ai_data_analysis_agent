import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # File Upload Settings
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    UPLOAD_FOLDER = 'data/uploads'
    TEMP_FOLDER = 'data/temp'
    
    # Data Processing Settings
    MAX_ROWS_DISPLAY = 1000
    SAMPLE_SIZE = 10000
    
    # AI Model Settings
    MODEL_NAME = "gpt-4"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.1
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)