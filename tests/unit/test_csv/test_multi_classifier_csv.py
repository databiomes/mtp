import pandas as pd
import pytest

from model_train_protocol import MultiClassifierInstruction, StateMachineInstruction
from model_train_protocol.common.constants import CSVType
from model_train_protocol.common.instructions.output.MultiClassifierOutput import parse_multi_classifier_output
from model_train_protocol.csv.conversion import CSVConversion, MultiCSVLine
from model_train_protocol.errors.conversion import ConversionError

def multi_dataframe():
    return pd.DataFrame({'Input': ['Hello', 'What should I wear?', 'Goodbye'], 'OutputOne': ['greeting', 'advice', 'farewell'], 'OutputTwo': ['positive', 'neutral', 'positive'], 'OutputThree': ['opening', 'question', 'closing'], 'Reference': ['A greeting', 'A request for advice', 'A farewell']})

def test_exact_legacy_shape_stays_single_output():
    dataframe = pd.DataFrame({'Input': ['one', 'two', 'three'], 'Output': ['a', 'b', 'c'], 'Reference': ['r1', 'r2', 'r3']})
    conversion = CSVConversion(dataframe)
    protocol = conversion.to_mtp()
    assert conversion.csv_type == CSVType.SINGLE_OUTPUT
    assert isinstance(next(iter(protocol.instructions)), StateMachineInstruction)

def test_extra_output_columns_select_multi_classifier():
    conversion = CSVConversion(multi_dataframe())
    assert conversion.csv_type == CSVType.MULTI_CLASSIFIER
    assert conversion.output_columns == ['OutputOne', 'OutputTwo', 'OutputThree']
    assert all(isinstance(line, MultiCSVLine) for line in conversion.ordered_lines)

def test_multi_classifier_protocol_contains_state_map_and_json_samples():
    instruction = next(iter(CSVConversion(multi_dataframe()).to_mtp().instructions))
    assert isinstance(instruction, MultiClassifierInstruction)
    assert instruction.get_states() == {'OutputOne': ['greeting', 'advice', 'farewell'], 'OutputTwo': ['positive', 'neutral'], 'OutputThree': ['opening', 'question', 'closing']}
    assert parse_multi_classifier_output(instruction.samples[0].output) == {'OutputOne': 'greeting', 'OutputTwo': 'positive', 'OutputThree': 'opening'}

def test_multi_classifier_output_inheritance_is_per_column():
    dataframe = pd.DataFrame({'Input': ['one', 'two', 'three'], 'OutputOne': ['a', '', 'c'], 'OutputTwo': ['x', 'y', ''], 'Reference': ['r1', 'r2', 'r3']})
    conversion = CSVConversion(dataframe)
    assert [line.outputs for line in conversion.ordered_lines] == [{'OutputOne': 'a', 'OutputTwo': 'x'}, {'OutputOne': 'a', 'OutputTwo': 'y'}, {'OutputOne': 'c', 'OutputTwo': 'y'}]

def test_missing_input_or_reference_is_rejected():
    with pytest.raises(ConversionError, match='Reference'):
        CSVConversion(pd.DataFrame({'Input': ['one'], 'OutputOne': ['a']}))

def test_no_output_columns_is_rejected():
    with pytest.raises(ConversionError, match='at least one output column'):
        CSVConversion(pd.DataFrame({'Input': ['one'], 'Reference': ['r']}))

def test_first_multi_output_row_must_be_complete():
    dataframe = pd.DataFrame({'Input': ['one', 'two', 'three'], 'OutputOne': ['', 'a', 'b'], 'OutputTwo': ['x', 'y', 'z'], 'Reference': ['r1', 'r2', 'r3']})
    with pytest.raises(ConversionError, match='OutputOne'):
        CSVConversion(dataframe)
