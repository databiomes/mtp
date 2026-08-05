from dataclasses import dataclass
from typing import List

import pandas as pd

from model_train_protocol import GuardrailError, StateMachineInstruction, StateMachineInput
from model_train_protocol.common.guardrails import Guardrail
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


class CSVConversion:
    """This class provides methods to convert from CSV into MTP"""
    instruction_name: str = "Output"
    input_col: str = "Input"
    output_col: str = "Output"
    context_col: str = "Reference"

    REQUIRED_COLUMNS: List[str] = [input_col, output_col, context_col]

    input_token: Token = Token("Input")
    input_tokenset: TokenSet = TokenSet(tokens=[input_token])

    def __init__(self, csv_data: pd.DataFrame, protocol_name: str = "CSV Protocol"):
        """
        Initializes the CSVConversion instance.

        :param csv_data: A dictionary where keys are column names and values are lists of column data.
        """
        self.csv_data: pd.DataFrame = self._process_dataframe(csv_data)
        self.ordered_lines: List[CSVLine] = self._format_lines()
        self.protocol: ProtocolV2 = ProtocolV2(name=protocol_name, inputs=1, encrypt=False)
        self.standard_input: StateMachineInput = StateMachineInput(
            tokensets=[self.input_tokenset])
        self.unique_states: set[str] = self._get_unique_states()

    def to_mtp(self) -> ProtocolV2:
        """Converts the CSV data to MTP format."""
        self._process_instruction()
        return self.protocol

    def _get_unique_states(self) -> set[str]:
        """
        Retrieves unique outputs from the CSV data.

        :return: A set of unique outputs.
        """
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
        dataframe = dataframe[~(dataframe[self.input_col].isna())]
        dataframe.reset_index(drop=True)
        return dataframe

    def _format_lines(self) -> List[CSVLine]:
        """
        Formats the DataFrame into a list of CSVLine objects.

        :return: A list of CSVLine objects containing the formatted data.
        """
        ordered_lines: List[CSVLine] = []
        previous_line: CSVLine | None = None
        for _, line in self.csv_data.iterrows():
            formatted_line: CSVLine = self._format_line(previous_line, line)
            ordered_lines.append(formatted_line)
            previous_line = formatted_line
        return ordered_lines

    def _format_line(self, previous_line: CSVLine | None, line: pd.Series) -> CSVLine:
        """
        Formats a single line of the DataFrame into a CSVLine object.

        :param line: A pandas Series representing a single line of the DataFrame.
        :return: A CSVLine object containing the formatted data.
        """
        output_str: str = str(line[self.output_col])

        if output_str == "nan" or output_str == "" or pd.isna(output_str):
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
            value: str = str(column[idx])
        except KeyError:
            return latest
        if value == "" or value == "nan" or pd.isna(value):
            return latest
        return value

    def _process_instruction(self) -> None:
        """
        Processes a single instruction and adds it to the protocol.
        """
        instruction_outputs: set[str] = self._get_unique_states()
        guardrail = Guardrail(
            good_prompt="Prompt related to the provided context of the model",
            bad_prompt="Prompt that is irrelevant and off topic",
            bad_output="GUARDRAIL"
        )

        instruction: StateMachineInstruction = StateMachineInstruction(
            input=self.standard_input, states=list(instruction_outputs)
        )

        for line in self.ordered_lines:

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
