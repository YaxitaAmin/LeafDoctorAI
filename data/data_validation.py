# src/data/data_validation.py
import tensorflow_data_validation as tfdv
from typing import Dict, Any
import pandas as pd

class DataValidator:
    def __init__(self, config):
        self.config = config
        self.schema = None
        
    def generate_schema(self, dataset):
        """Generate schema from training data"""
        stats = tfdv.generate_statistics_from_dataframe(dataset)
        self.schema = tfdv.infer_schema(stats)
        tfdv.write_schema_text(self.schema, 'schema.pbtxt')
        
    def validate_dataset(self, dataset):
        """Validate dataset against schema"""
        stats = tfdv.generate_statistics_from_dataframe(dataset)
        anomalies = tfdv.validate_statistics(stats, schema=self.schema)
        return anomalies
        
    def check_data_drift(self, reference_data, current_data):
        """Check for data drift between two datasets"""
        ref_stats = tfdv.generate_statistics_from_dataframe(reference_data)
        current_stats = tfdv.generate_statistics_from_dataframe(current_data)
        drift_anomalies = tfdv.validate_statistics(
            statistics=current_stats,
            schema=self.schema,
            previous_statistics=ref_stats
        )
        return drift_anomalies