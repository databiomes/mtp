import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import List

import pandas as pd

from model_train_protocol import (
    GuardrailError,
    MultiClassifierInstruction,
    StateMachineInstruction,
    StateMachineInput,
)
from model_train_protocol.common.guardrails import Guardrail
from model_train_protocol.common.constants import CSVType
from model_train_protocol.common.instructions.output.MultiClassifierOutput import format_multi_classifier_output
from model_train_protocol.common.tokens import Token, TokenSet
from model_train_protocol.errors.conversion import ConversionError
from model_train_protocol.v2 import ProtocolV2


@dataclass
class CSVLine:
    """This class represents a single line in the CSV data."""
    input_str: str
    output_str: str
    context_str: str

    @property
    def is_guardrail(self) -> bool:
        """Determines if the line is a guardrail based on its content."""
        return self.output_str == "GUARDRAIL"


@dataclass
class MultiCSVLine:
    """This class represents a single line in a multi-output CSV."""
    input_str: str
    outputs: dict[str, str]
    context_str: str


def read_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Reads the content of a CSV file into a dataframe.

    Column names are read before the file is parsed, because pandas renames a
    repeated header as it reads, so a file that declares Output twice would
    otherwise arrive as the columns Output and Output.1 and be converted as
    though it declared a second output column.

    :param file_bytes: The raw content of the CSV file.
    :return: The data in the CSV file.
    :raises ConversionError: If a column name appears more than once.
    """
    duplicates = _get_duplicate_columns(file_bytes)
    if duplicates:
        repeated = ", ".join(repr(column) for column in duplicates)
        raise ConversionError(f"CSV cannot contain duplicate column names: {repeated}.")
    return pd.read_csv(BytesIO(file_bytes))


def _get_duplicate_columns(file_bytes: bytes) -> List[str]:
    """
    Finds the column names that a CSV header uses more than once.

    :param file_bytes: The raw content of the CSV file.
    :return: The repeated names, in the order they are first seen.
    """
    seen: set[str] = set()
    duplicates: List[str] = []
    for column in _get_header_columns(file_bytes):
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def _get_header_columns(file_bytes: bytes) -> List[str]:
    """
    Reads the column names from the first non-empty line of a CSV.

    :param file_bytes: The raw content of the CSV file.
    :return: The column names, or an empty list if the file is not UTF-8 text.
    """
    try:
        content = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return []
    for row in csv.reader(StringIO(content, newline="")):
        if any(row):
            return row
    return []


class CSVConversion:
    """This class provides methods to convert from CSV into MTP"""
    instruction_name: str = "Output"
    input_col: str = "Input"
    output_col: str = "Output"
    context_col: str = "Reference"

    REQUIRED_COLUMNS: List[str] = [input_col, output_col, context_col]
    BASE_REQUIRED_COLUMNS: List[str] = [input_col, context_col]

    input_token: Token = Token("Input")
    input_tokenset: TokenSet = TokenSet(tokens=[input_token])

    def __init__(self, csv_data: pd.DataFrame, protocol_name: str = "CSV Protocol"):
        """
        Initializes the CSVConversion instance.

        :param csv_data: A dictionary where keys are column names and values are lists of column data.
        """
        self._validate_columns(csv_data)
        self.output_columns: List[str] = self._get_output_columns(csv_data)
        self.csv_type: CSVType = self._get_csv_type(csv_data)
        self.csv_data: pd.DataFrame = self._process_dataframe(csv_data)
        self.ordered_lines: List[CSVLine | MultiCSVLine] = (
            self._format_multi_lines() if self.csv_type == CSVType.MULTI_CLASSIFIER else self._format_lines()
        )
        self.protocol: ProtocolV2 = ProtocolV2(name=protocol_name, inputs=1, encrypt=False)
        self.standard_input: StateMachineInput = StateMachineInput(
            tokensets=[self.input_tokenset])
        if self.csv_type == CSVType.MULTI_CLASSIFIER:
            self.multi_classifier_state_map: dict[str, list[str]] = self._get_multi_classifier_state_map()
        elif self.csv_type == CSVType.SINGLE_OUTPUT:
            self.unique_states: set[str] = self._get_single_output_states()
        else:
            raise ConversionError(f"Unsupported CSV type: {self.csv_type}")

    def to_mtp(self) -> ProtocolV2:
        """Converts the CSV data to MTP format."""
        if self.csv_type == CSVType.MULTI_CLASSIFIER:
            self._process_multi_instruction()
        else:
            self._process_instruction()
        return self.protocol

    @classmethod
    def _validate_columns(cls, dataframe: pd.DataFrame) -> None:
        columns = list(dataframe.columns)
        missing_columns = [column for column in cls.BASE_REQUIRED_COLUMNS if column not in columns]
        if missing_columns:
            raise ConversionError(f"CSV is missing required columns: {', '.join(missing_columns)}.")
        if len(columns) != len(set(columns)):
            raise ConversionError("CSV cannot contain duplicate column names.")
        if not [column for column in columns if column not in cls.BASE_REQUIRED_COLUMNS]:
            raise ConversionError("CSV must contain at least one output column in addition to Input and Reference.")

    @classmethod
    def _get_output_columns(cls, dataframe: pd.DataFrame) -> List[str]:
        return [column for column in dataframe.columns if column not in cls.BASE_REQUIRED_COLUMNS]

    @classmethod
    def _is_single_output_shape(cls, dataframe: pd.DataFrame) -> bool:
        return set(dataframe.columns) == set(cls.REQUIRED_COLUMNS) and len(dataframe.columns) == len(cls.REQUIRED_COLUMNS)

    @classmethod
    def _get_csv_type(cls, dataframe: pd.DataFrame) -> CSVType:
        return CSVType.SINGLE_OUTPUT if cls._is_single_output_shape(dataframe) else CSVType.MULTI_CLASSIFIER

    def _get_multi_classifier_state_map(self) -> dict[str, list[str]]:
        states: dict[str, list[str]] = {column: [] for column in self.output_columns}
        for line in self.ordered_lines:
            for column, value in line.outputs.items():
                if value not in states[column]:
                    states[column].append(value)
        return states


    def _get_single_output_states(self) -> set[str]:
        unique_outputs: set[str] = set()
        for line in self.ordered_lines:
            if line.output_str != "" and not pd.isna(line.output_str):
                unique_outputs.add(line.output_str)
        if "GUARDRAIL" in unique_outputs:
            unique_outputs.remove("GUARDRAIL")
        return unique_outputs

    def _process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame to match expected format.

        :param dataframe: The input DataFrame.
        :return: Processed DataFrame
        """
        # Remove rows where Input is empty
        dataframe = dataframe[~(dataframe[self.input_col].isna())].copy()
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    @staticmethod
    def _is_empty(value: object) -> bool:
        if value is None or pd.isna(value):
            return True
        return str(value) == ""

    def _format_lines(self) -> List[CSVLine]:
        """
        Formats the DataFrame into a list of CSVLine objects.

        :return: A list of CSVLine objects containing the formatted data.
        """
        ordered_lines: List[CSVLine] = []
        previous_output: str | None = None
        for input_value, output_value, context_value in self.csv_data[
            [self.input_col, self.output_col, self.context_col]
        ].itertuples(index=False, name=None):
            if self._is_empty(output_value):
                if previous_output is None:
                    raise ConversionError("The first line of the CSV cannot have an empty output.")
                output_str = previous_output
            else:
                output_str = str(output_value)

            formatted_line = CSVLine(
                input_str=str(input_value),
                output_str=output_str,
                context_str=str(context_value),
            )
            ordered_lines.append(formatted_line)
            previous_output = output_str
        return ordered_lines

    def _format_multi_lines(self) -> List[MultiCSVLine]:
        ordered_lines: List[MultiCSVLine] = []
        previous_outputs: dict[str, str | None] = {column: None for column in self.output_columns}

        for _, row in self.csv_data.iterrows():
            outputs: dict[str, str] = {}
            for column in self.output_columns:
                value = row[column]
                if self._is_empty(value):
                    previous_value = previous_outputs[column]
                    if previous_value is None:
                        raise ConversionError(
                            f"The first line of the CSV cannot have an empty output in column '{column}'."
                        )
                    value_str = previous_value
                else:
                    value_str = str(value)
                outputs[column] = value_str
                previous_outputs[column] = value_str

            ordered_lines.append(
                MultiCSVLine(
                    input_str=str(row[self.input_col]),
                    outputs=outputs,
                    context_str=str(row[self.context_col]),
                )
            )
        return ordered_lines

    def _format_line(self, previous_line: CSVLine | None, line: pd.Series) -> CSVLine:
        """
        Formats a single line of the DataFrame into a CSVLine object.

        :param line: A pandas Series representing a single line of the DataFrame.
        :return: A CSVLine object containing the formatted data.
        """
        output_str: str = str(line[self.output_col])

        if self._is_empty(output_str):
            if previous_line is None:
                raise ConversionError("The first line of the CSV cannot have an empty output.")
            output_str = str(previous_line.output_str)

        return CSVLine(
            input_str=str(line[self.input_col]),
            output_str=output_str,
            context_str=str(line[self.context_col])
        )

    @classmethod
    def _assign_latest(cls, column: pd.Series, idx: int, latest: str) -> str:
        """Helper function to assign latest non-empty value."""
        try:
            value = column.iloc[idx]
        except (IndexError, KeyError):
            return latest
        return latest if cls._is_empty(value) else str(value)

    def _process_instruction(self) -> None:
        """
        Processes a single instruction and adds it to the protocol.
        """
        instruction_outputs: set[str] = self.unique_states
        guardrail = Guardrail(
            good_prompt="Prompt related to the provided context of the model",
            bad_prompt="Prompt that is irrelevant and off topic",
            bad_output="GUARDRAIL"
        )

        instruction: StateMachineInstruction = StateMachineInstruction(
            input=self.standard_input, states=list(instruction_outputs)
        )

        for line in self.ordered_lines:
            assert isinstance(line, CSVLine)

            if line.is_guardrail:
                guardrail.add_sample(line.input_str)
                continue

            if line.context_str != "" and line.context_str != "nan" and not pd.isna(line.context_str):
                instruction.add_context(line.context_str)

            instruction.add_sample(
                input_snippets=[line.input_str],
                state=line.output_str
            )

        if instruction.has_guardrails and 0 < len(guardrail.samples) < 3:
            raise GuardrailError(
                "At least 3 guardrail samples are required. Please add more guardrail samples to the CSV data.")
        elif len(guardrail.samples) >= 3:
            instruction.add_guardrail(guardrail=guardrail, tokenset_index=0)

        self.protocol.add_instruction(instruction)

    def _process_multi_instruction(self) -> None:
        """Create a MultiClassifierInstruction from the output columns."""
        instruction = MultiClassifierInstruction(input=self.standard_input, state_map=self.multi_classifier_state_map)

        for line in self.ordered_lines:
            assert isinstance(line, MultiCSVLine)
            if line.context_str != "" and line.context_str != "nan" and not pd.isna(line.context_str):
                instruction.add_context(line.context_str)
            instruction.add_sample(
                input_snippets=[line.input_str],
                output_snippet=format_multi_classifier_output(line.outputs),
            )

        self.protocol.add_instruction(instruction)
