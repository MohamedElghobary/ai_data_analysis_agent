import unittest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Customer_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'department': np.random.choice(['Sales', 'Marketing', 'Engineering'], 100),
            'missing_col': [np.nan if i % 10 == 0 else f'value_{i}' for i in range(100)]
        })
        self.processor = DataProcessor(self.sample_data)
    
    def test_get_column_info(self):
        column_info = self.processor.get_column_info()
        self.assertEqual(len(column_info), len(self.sample_data.columns))
        self.assertIn('Column', column_info.columns)
        self.assertIn('Data Type', column_info.columns)
    
    def test_analyze_missing_data(self):
        missing_data = self.processor.analyze_missing_data()
        self.assertGreater(len(missing_data), 0)  # Should have missing data
        self.assertIn('missing_col', missing_data['Column'].values)
    
    def test_basic_statistics(self):
        stats = self.processor.basic_statistics()
        self.assertIn('age', stats.columns)
        self.assertIn('salary', stats.columns)
    
    def test_correlation_analysis(self):
        corr = self.processor.correlation_analysis()
        self.assertEqual(len(corr), len(self.sample_data.select_dtypes(include=[np.number]).columns))

if __name__ == '__main__':
    unittest.main()