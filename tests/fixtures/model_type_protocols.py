"""
Builders for a minimal, valid protocol of each model type.

Used by the tests that compare how the three model types behave: the templates they produce, and the rule that a
protocol may not mix instructions from more than one of them.
"""
import json

import model_train_protocol as mtp

# "amused" is deliberately never used by a sample, so tests can tell a declared state map apart from one rebuilt out of
# the samples.
STATE_MAP = {
    "emotion": ["curious", "afraid", "confused", "amused"],
    "intent": ["question", "statement"],
}

MULTI_CLASSIFIER_SAMPLES = [
    ("What a curious feeling, I must be shutting up like a telescope!", {"emotion": "curious", "intent": "statement"}),
    ("Which way ought I to go from here?", {"emotion": "confused", "intent": "question"}),
    ("Oh dear, I do hope this fall will ever come to an end.", {"emotion": "afraid", "intent": "statement"}),
    ("How funny it seems to talk to a cat that keeps vanishing!", {"emotion": "curious", "intent": "statement"}),
    ("Are you quite certain that everyone here is mad?", {"emotion": "confused", "intent": "question"}),
]

STATE_MACHINE_STATES = ["QUESTION", "STATEMENT"]


def input_tokenset(action: str = "Talk") -> mtp.TokenSet:
    """Builds a single input TokenSet. The action varies so a protocol can hold two distinct instructions."""
    return mtp.TokenSet(tokens=(mtp.Token("Tree"), mtp.Token("English"), mtp.Token("Alice"), mtp.Token(action)))


def build_multi_classifier_instruction(state_map: dict[str, list[str]] | None = None,
                                       action: str = "Talk",
                                       samples: list[tuple[str, dict[str, str]]] | None = None
                                       ) -> mtp.MultiClassifierInstruction:
    """
    Builds a sampled MultiClassifierInstruction over the given state map.

    :param state_map: Classification labels mapped to their acceptable values. Defaults to STATE_MAP.
    :param action: Varies the input TokenSet so two instructions can live in the same protocol.
    :param samples: (input line, classification) pairs. Must carry exactly the state map's keys. Defaults to
        MULTI_CLASSIFIER_SAMPLES, which matches STATE_MAP.
    """
    instruction = mtp.MultiClassifierInstruction(
        input=mtp.InstructionInput(tokensets=[input_tokenset(action)]),
        state_map=STATE_MAP if state_map is None else state_map,
        context=["Each response is a JSON object with exactly the keys of the state map."],
    )
    for line, classification in (MULTI_CLASSIFIER_SAMPLES if samples is None else samples):
        instruction.add_sample(input_snippets=[line], output_snippet=json.dumps(classification))
    return instruction


def build_multi_classifier_protocol() -> mtp.Protocol:
    """Builds a minimal, valid multi classifier protocol."""
    protocol = mtp.Protocol(name="multi_classifier_template_test", inputs=1, encrypt=False)
    protocol.add_context("The Cheshire Cat classifies each line Alice speaks.")
    protocol.add_instruction(build_multi_classifier_instruction())
    return protocol


def build_state_machine_instruction(action: str = "Talk") -> mtp.StateMachineInstruction:
    """Builds a sampled StateMachineInstruction."""
    instruction = mtp.StateMachineInstruction(
        input=mtp.StateMachineInput(tokensets=[input_tokenset(action)]),
        states=STATE_MACHINE_STATES,
    )
    for line, classification in MULTI_CLASSIFIER_SAMPLES:
        instruction.add_sample(input_snippets=[line], state=classification["intent"].upper())
    return instruction


def build_state_machine_protocol() -> mtp.Protocol:
    """Builds a minimal, valid state machine protocol."""
    protocol = mtp.Protocol(name="state_machine_template_test", inputs=1, encrypt=False)
    protocol.add_context("The Cheshire Cat labels each line Alice speaks.")
    protocol.add_instruction(build_state_machine_instruction())
    return protocol


def build_generative_instruction(action: str = "Talk", name: str = "alice_cat_reply") -> mtp.Instruction:
    """Builds a sampled basic (generative) Instruction."""
    reply_token: mtp.FinalToken = mtp.FinalToken(f"Continue{action}", desc="The Cat keeps the conversation going.")
    instruction = mtp.Instruction(
        input=mtp.InstructionInput(tokensets=[input_tokenset(action)]),
        output=mtp.InstructionOutput(tokenset=input_tokenset("Reply"), final=[reply_token]),
        context=["The Cat replies to Alice."],
        name=name,
    )
    for line, _ in MULTI_CLASSIFIER_SAMPLES:
        instruction.add_sample(input_snippets=[line], output_snippet="Everyone here is quite mad, you know.",
                               final=reply_token)
    return instruction


def build_extended_instruction(action: str = "Ponder", name: str = "alice_cat_extended") -> mtp.ExtendedInstruction:
    """Builds a sampled ExtendedInstruction, which is generative like a basic Instruction."""
    reply_token: mtp.FinalToken = mtp.FinalToken(f"Continue{action}", desc="The Cat keeps the conversation going.")
    context_tokenset: mtp.TokenSet = input_tokenset(action)
    prompt_tokenset: mtp.TokenSet = input_tokenset("Prompt")
    instruction = mtp.ExtendedInstruction(
        input=mtp.InstructionInput(tokensets=[context_tokenset, prompt_tokenset]),
        output=mtp.ExtendedResponse(final=reply_token),
        context=["The Cat replies to Alice at length."],
        name=name,
    )
    for line, _ in MULTI_CLASSIFIER_SAMPLES:
        instruction.add_sample(
            inputs=[
                context_tokenset.create_snippet(string=line),
                prompt_tokenset.create_snippet(string="Why is a raven like a writing desk?"),
            ],
            response_string="Everyone here is quite mad, you know.",
            final=reply_token,
        )
    return instruction


def build_generative_protocol() -> mtp.Protocol:
    """Builds a minimal, valid generative protocol."""
    protocol = mtp.Protocol(name="generative_template_test", inputs=1, encrypt=False)
    protocol.add_context("The Cheshire Cat answers each line Alice speaks.")
    protocol.add_instruction(build_generative_instruction())
    return protocol
