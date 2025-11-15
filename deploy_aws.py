"""
AWS Deployment Script for EPL Predictor
"""

import boto3
import zipfile
import os
import json

def create_lambda_package():
    """Create deployment package for AWS Lambda"""
    
    # Files to include in Lambda package
    files_to_include = [
        'app.py',
        'src/',
        'templates/',
        'static/',
        'models/',
        'data/epl_combined.csv',
        'outputs/'
    ]
    
    with zipfile.ZipFile('epl_predictor_lambda.zip', 'w') as zipf:
        for item in files_to_include:
            if os.path.isfile(item):
                zipf.write(item)
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path)
    
    print("Lambda package created: epl_predictor_lambda.zip")

def deploy_to_ec2():
    """Deploy to EC2 instance"""
    
    user_data_script = """#!/bin/bash
    yum update -y
    yum install -y python3 python3-pip git
    
    # Clone your repository (replace with your repo URL)
    git clone https://github.com/yourusername/epl-predictor.git /home/ec2-user/epl-predictor
    cd /home/ec2-user/epl-predictor
    
    # Install dependencies
    pip3 install -r requirements.txt
    
    # Start the application
    python3 app.py
    """
    
    print("EC2 User Data Script:")
    print(user_data_script)

if __name__ == '__main__':
    create_lambda_package()
    deploy_to_ec2()