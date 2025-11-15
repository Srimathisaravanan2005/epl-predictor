"""
JSON Serialization Utilities for EPL Prediction Application
Handles numpy types and other non-serializable objects
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import jsonify

class EPLJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for EPL prediction data"""
    
    def default(self, obj):
        """Convert non-serializable objects to serializable format"""
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle pandas types
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
        
        # Let the base class handle other types
        return super().default(obj)

def safe_jsonify(data, **kwargs):
    """
    Safely convert data to JSON response, handling numpy types
    
    Args:
        data: Data to serialize
        **kwargs: Additional arguments for jsonify
        
    Returns:
        Flask JSON response
    """
    try:
        # Convert numpy types recursively
        clean_data = convert_for_json(data)
        
        # Create Flask response
        response = jsonify(clean_data, **kwargs)
        return response
        
    except (TypeError, ValueError) as e:
        # Fallback error response
        error_response = {
            'error': 'JSON serialization failed',
            'message': str(e),
            'data_type': str(type(data).__name__)
        }
        return jsonify(error_response), 500

def convert_for_json(obj):
    """
    Recursively convert objects to JSON-serializable format
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if obj is None:
        return None
    
    # Handle numpy types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        return {str(k): convert_for_json(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, pd.DataFrame):
        return [convert_for_json(row) for row in obj.to_dict('records')]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    # Handle datetime objects
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Handle collections
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return [convert_for_json(item) for item in obj]
    
    # Handle basic types
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Try to convert other objects to string
    else:
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__} object>"

def validate_prediction_response(prediction, probabilities):
    """
    Validate and clean prediction response data
    
    Args:
        prediction: Model prediction result
        probabilities: Model probability outputs
        
    Returns:
        tuple: (cleaned_prediction, cleaned_probabilities)
    """
    try:
        # Clean prediction
        clean_prediction = convert_for_json(prediction)
        
        # Clean probabilities
        clean_probabilities = {}
        if probabilities:
            for key, value in probabilities.items():
                clean_key = str(key)
                clean_value = float(value) if isinstance(value, (np.floating, float)) else value
                clean_probabilities[clean_key] = clean_value
        
        return clean_prediction, clean_probabilities
        
    except Exception as e:
        print(f"Error validating prediction response: {e}")
        return None, {}

def create_prediction_response(prediction, probabilities, input_data=None, metadata=None):
    """
    Create a standardized prediction response
    
    Args:
        prediction: Model prediction
        probabilities: Prediction probabilities
        input_data: Input data used for prediction
        metadata: Additional metadata
        
    Returns:
        dict: Standardized response
    """
    # Clean the data
    clean_prediction, clean_probabilities = validate_prediction_response(prediction, probabilities)
    
    response = {
        'success': True,
        'prediction': clean_prediction,
        'probabilities': clean_probabilities,
        'timestamp': datetime.now().isoformat()
    }
    
    if input_data:
        response['input_data'] = convert_for_json(input_data)
    
    if metadata:
        response['metadata'] = convert_for_json(metadata)
    
    return response

def create_error_response(error_message, error_code=None, details=None):
    """
    Create a standardized error response
    
    Args:
        error_message: Error message
        error_code: Optional error code
        details: Additional error details
        
    Returns:
        dict: Standardized error response
    """
    response = {
        'success': False,
        'error': str(error_message),
        'timestamp': datetime.now().isoformat()
    }
    
    if error_code:
        response['error_code'] = error_code
    
    if details:
        response['details'] = convert_for_json(details)
    
    return response

# AWS Lambda-specific utilities
def lambda_response(status_code, body, headers=None):
    """
    Create AWS Lambda-compatible response
    
    Args:
        status_code: HTTP status code
        body: Response body
        headers: Optional headers
        
    Returns:
        dict: Lambda response format
    """
    default_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
    }
    
    if headers:
        default_headers.update(headers)
    
    return {
        'statusCode': status_code,
        'headers': default_headers,
        'body': json.dumps(convert_for_json(body), cls=EPLJSONEncoder)
    }

def lambda_success_response(data, message="Success"):
    """Create successful Lambda response"""
    body = {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    return lambda_response(200, body)

def lambda_error_response(error_message, status_code=400, error_code=None):
    """Create error Lambda response"""
    body = create_error_response(error_message, error_code)
    return lambda_response(status_code, body)

# Best practices for AWS deployment
class AWSJSONUtils:
    """Utilities specifically for AWS deployment"""
    
    @staticmethod
    def prepare_for_dynamodb(data):
        """Prepare data for DynamoDB storage"""
        # DynamoDB doesn't support empty strings or None values in certain contexts
        def clean_for_dynamo(obj):
            if isinstance(obj, dict):
                return {k: clean_for_dynamo(v) for k, v in obj.items() if v is not None and v != ""}
            elif isinstance(obj, list):
                return [clean_for_dynamo(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        return clean_for_dynamo(convert_for_json(data))
    
    @staticmethod
    def prepare_for_s3(data):
        """Prepare data for S3 storage"""
        return json.dumps(convert_for_json(data), cls=EPLJSONEncoder, indent=2)
    
    @staticmethod
    def prepare_for_api_gateway(data):
        """Prepare data for API Gateway response"""
        return convert_for_json(data)