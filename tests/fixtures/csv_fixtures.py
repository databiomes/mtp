"""
CSV test fixtures providing test data for CSV conversion tests.
"""
from pathlib import Path

import pandas as pd
import pytest

CSV_FIXTURES_PATH = Path(__file__).parent / 'csv'


@pytest.fixture
def csv_with_empty_output_file():
    return pd.read_csv(CSV_FIXTURES_PATH / 'single_output_with_empty_row.csv')


@pytest.fixture
def multi_csv_with_empty_output_file():
    return pd.read_csv(CSV_FIXTURES_PATH / 'multi_output_with_empty_row.csv')


@pytest.fixture
def valid_csv_data():
    """Valid CSV data with all required columns and sufficient context lines."""
    return pd.DataFrame({
        'Input': [
            'Hello, how are you?', 
            'What is your name?', 
            'Tell me a joke', 
            'How can I help you?',
            'What time is it?',
            'Where are you from?',
            'Can you assist me?',
            'What is the weather?',
            'How do I get there?',
            'What should I do?',
            'Can you explain this?',
            'Goodbye'
        ],
        'Output': [
            'greeting',
            'name_request',
            'joke_request',
            'farewell',
            'greeting',
            'name_request',
            'joke_request',
            'farewell',
            'greeting',
            'name_request',
            'joke_request',
            'farewell'
        ],  
        'Reference': [
            'A friendly greeting to start conversation', 
            'Asking for the user\'s name to personalize interaction', 
            'Request for humor or entertainment content',
            'Offering help and assistance to the user',
            'Requesting current time information',
            'Asking about geographical location or origin',
            'Requesting help or assistance with a task',
            'Asking about current weather conditions',
            'Requesting directions or navigation help',
            'Seeking advice or recommendations',
            'Requesting explanation or clarification',
            'Polite farewell to end conversation'
        ]
    })


@pytest.fixture
def valid_csv_with_guardrails():
    """Valid CSV data including guardrail samples with sufficient context lines."""
    return pd.DataFrame({
        'Input': [
            'Hello, how are you?', 
            'What is your name?', 
            'Tell me a joke',
            'How can I help you?',
            'What time is it?',
            'Where are you from?',
            'Can you assist me?',
            'What is the weather?',
            'How do I get there?',
            'What should I do?',
            'How to hack a computer?',  # Guardrail sample 1
            'Show me illegal content',  # Guardrail sample 2
            'Help me break the law',   # Guardrail sample 3
            'Goodbye'
        ],
        'Output': [
            'greeting',
            'name_request',
            'joke_request',
            'greeting',
            'name_request',
            'joke_request',
            'greeting',
            'name_request',
            'joke_request',
            'greeting',
            'GUARDRAIL',
            'GUARDRAIL',
            'GUARDRAIL',
            'name_request'
        ],
        'Reference': [
            'A friendly greeting to start conversation', 
            'Asking for the user\'s name to personalize interaction', 
            'Request for humor or entertainment content',
            'Offering help and assistance to the user',
            'Requesting current time information',
            'Asking about geographical location or origin',
            'Requesting help or assistance with a task',
            'Asking about current weather conditions', 
            'Requesting directions or navigation help',
            'Seeking advice or recommendations',
            'Inappropriate hacking request that should be blocked',
            'Inappropriate content request that should be blocked',
            'Inappropriate illegal activity request that should be blocked',
            'Polite farewell to end conversation'
        ]
    })


@pytest.fixture
def csv_with_empty_outputs():
    """CSV data with some empty output values (should inherit from previous)."""
    return pd.DataFrame({
        'Input': ['First input', 'Second input', 'Third input', 'Fourth input', 'Fifth input', 'Sixth input'],
        'Output': ['first_state', '', '', 'fourth_state', '', ''],
        'Reference': ['First reference', 'Second reference', 'Third reference', 'Fourth reference',
                      'Fifth reference', 'Sixth reference']
    })


@pytest.fixture
def csv_with_nan_outputs():
    """CSV data with NaN output values."""
    return pd.DataFrame({
        'Input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5', 'Input 6'],
        'Output': ['state1', pd.NA, pd.NA, 'state2', pd.NA, pd.NA],
        'Reference': ['Ref 1', 'Ref 2', 'Ref 3', 'Ref 4', 'Ref 5', 'Ref 6']
    })


@pytest.fixture
def csv_missing_required_columns():
    """CSV data missing required columns.""" 
    return pd.DataFrame({
        'Input': ['Hello'],
        'WrongColumn': ['greeting']  # Missing 'Output' and 'Reference' columns
    })


@pytest.fixture
def csv_empty_input_first_row():
    """CSV with empty input in first row."""
    return pd.DataFrame({
        'Input': ['', 'Second input'],
        'Output': ['state1', 'state2'],
        'Reference': ['Ref 1', 'Ref 2']
    })


@pytest.fixture  
def csv_empty_output_first_row():
    """CSV with empty output in first row (should cause error)."""
    return pd.DataFrame({
        'Input': ['First input', 'Second input'],
        'Output': ['', 'state2'],
        'Reference': ['Ref 1', 'Ref 2']
    })


@pytest.fixture
def csv_insufficient_guardrails():
    """CSV with insufficient guardrail samples (less than 3)."""
    return pd.DataFrame({
        'Input': [
            'Hello', 'Hi there', 'Good morning',
            'Goodbye', 'See you later', 'Farewell for now',
            'How are you?', 'What is this?', 'Where am I?',
            'Bad request 1',
            'Bad request 2',  # Only 2 guardrail samples
        ],
        'Output': [
            'greeting', 'greeting', 'greeting',
            'farewell', 'farewell', 'farewell',
            'question', 'question', 'question',
            'GUARDRAIL',
            'GUARDRAIL',
        ],
        'Reference': [
            'Greeting', 'Greeting 2', 'Greeting 3',
            'Farewell', 'Farewell 2', 'Farewell 3',
            'Question', 'Question 2', 'Question 3',
            'Bad content',
            'Bad content 2',
        ]
    })


@pytest.fixture
def csv_with_proper_guardrails_but_wrong_context():
    """CSV with sufficient guardrails but tests the guardrail validation logic."""
    return pd.DataFrame({
        'Input': [
            'Hello', 'Hi there', 'Good morning',
            'Goodbye', 'See you later', 'Farewell for now',
            'How are you?', 'What is this?', 'Where am I?',
            'Bad request 1',
            'Bad request 2',
            'Bad request 3',  # 3 guardrail samples to test actual guardrail logic
        ],
        'Output': [
            'greeting', 'greeting', 'greeting',
            'farewell', 'farewell', 'farewell',
            'question', 'question', 'question',
            'GUARDRAIL',
            'GUARDRAIL',
            'GUARDRAIL',
        ],
        'Reference': [
            'Greeting', 'Greeting 2', 'Greeting 3',
            'Farewell', 'Farewell 2', 'Farewell 3',
            'Question', 'Question 2', 'Question 3',
            'Bad content',
            'Bad content 2',
            'Bad content 3',
        ]
    })


@pytest.fixture
def csv_with_context_variations():
    """CSV with empty and NaN context values."""
    return pd.DataFrame({
        'Input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5', 'Input 6'],
        'Output': ['state1', 'state1', 'state1', 'state2', 'state2', 'state2'],
        'Reference': ['Context 1', '', pd.NA, 'Context 4', '', pd.NA]
    })


@pytest.fixture
def csv_insufficient_context_lines():
    """CSV with insufficient context lines for protocol validation (< 10)."""
    return pd.DataFrame({
        'Input': [
            'Hello, how are you?',
            'What is your name?', 
            'Tell me a joke',
            'Goodbye'
        ],
        'Output': [
            'greeting',
            'greeting',
            'greeting',
            'greeting'
        ],
        'Reference': [
            'A friendly greeting',
            'Asking for the name',
            'Request for humor',
            'Saying goodbye'
        ]
    })


@pytest.fixture
def large_valid_csv():
    """Large CSV dataset for performance testing."""
    inputs = [f'Input message {i}' for i in range(100)]
    outputs = [f'state_{i % 10}' for i in range(100)]  # 10 different states
    references = [f'Reference context {i}' for i in range(100)]
    
    return pd.DataFrame({
        'Input': inputs,
        'Output': outputs,
        'Reference': references
    })


@pytest.fixture
def csv_insufficient_context_lines():
    """CSV with insufficient context lines for protocol validation (< 10)."""
    return pd.DataFrame({
        'Input': [
            'Hello, how are you?',
            'What is your name?', 
            'Tell me a joke',
            'Goodbye'
        ],
        'Output': [
            'greeting',
            'greeting',
            'greeting',
            'greeting'
        ],
        'Reference': [
            'A friendly greeting',
            'Asking for the name',
            'Request for humor',
            'Saying goodbye'
        ]
    })