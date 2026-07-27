import json

import model_train_protocol as mtp

# Cheshire Cat Line Classifier

# This example protocol demonstrates a MultiClassifierInstruction. Instead of generating a free-form reply, the model
# reads a single line spoken by Alice (from "Alice's Adventures in Wonderland") and classifies it along two independent
# dimensions at once: the "emotion" Alice is expressing and the "intent" of her line.
#
# A MultiClassifierInstruction responds with a JSON object whose keys are the classification labels defined in the
# state_map, and whose values are one of the acceptable options for that label. Every response must contain exactly the
# keys defined in the state_map — no more, no fewer.

protocol = mtp.Protocol(name="multi_classifier_example", inputs=1, encrypt=False)

protocol.add_context(
    "Alice was tired of sitting by her sister with nothing to do, and she began to wonder what she could do to pass the time.")
protocol.add_context(
    "The hot day made her sleepy, when suddenly a white rabbit with pink eyes ran past her, muttering about being late.")
protocol.add_context(
    "Burning with curiosity, Alice chased the rabbit down a rabbit-hole and fell for a very long time before landing softly.")
protocol.add_context(
    "She found herself in a long, low hall, wandering through a strange world where nothing behaved quite as it should.")
protocol.add_context(
    "In a tree, surrounded by a dense fog, the Cheshire Cat watches Alice and listens closely to everything she says.")
protocol.add_context(
    "The Cat does not reply with words. Instead, it silently reads each thing Alice says and judges her mood and purpose.")
protocol.add_context(
    "Alice's lines swing between wonder, fear, and confusion as she tries to make sense of Wonderland.")
protocol.add_context(
    "Sometimes she asks a question, sometimes she simply states what she sees, and sometimes she cries out in surprise.")
protocol.add_context(
    "The Cat's task is to label each line Alice speaks by the emotion behind it and the intent it carries.")
protocol.add_context(
    "Every judgement is recorded as a small JSON note containing exactly one emotion and one intent.")

# -------------------- Setup Tokens ---------------------

# Language
token_english: mtp.Token = mtp.Token("English")

# Characters
token_alice: mtp.Token = mtp.Token("Alice")

# Scenes
token_tree: mtp.Token = mtp.Token("Tree",
                                  desc="Perched in a tree, surrounded by a dense fog where nothing can be seen past a few feet, the Cheshire Cat sits smiling on a branch.")

# Actions
token_talk: mtp.Token = mtp.Token("Talk")

# Create the input TokenSet: a single line spoken by Alice, in English, within the Tree scene.
tree_english_alice_talk: mtp.TokenSet = mtp.TokenSet(tokens=(token_tree, token_english, token_alice, token_talk))

# -------------------- Multi Classifier Instruction --------------------

# The state_map defines the classification labels (keys) and their acceptable values (lists of strings).
# This instruction has two classifications: "emotion" and "intent".
state_map = {
    "emotion": ["curious", "afraid", "confused", "amused"],
    "intent": ["question", "statement", "exclamation"],
}

# Construct the Input format. A MultiClassifier accepts one or more input TokenSets; here we use a single line.
alice_line_input: mtp.InstructionInput = mtp.InstructionInput(
    tokensets=[tree_english_alice_talk],
)

# Create the Instruction. The output format (a JSON classification) is derived automatically from the state_map,
# so we only need to provide the input and the state_map. Background context is optional.
alice_line_classifier: mtp.MultiClassifierInstruction = mtp.MultiClassifierInstruction(
    input=alice_line_input,
    state_map=state_map,
    context=[
        "The Cheshire Cat classifies each line Alice speaks by the emotion she expresses and the intent of the line.",
        "Each response is a JSON object with exactly the keys 'emotion' and 'intent'.",
    ],
)

# Add at least 5 samples. Each output snippet must be valid JSON containing exactly the state_map keys.

# 1st Sample
alice_line_classifier.add_sample(
    input_snippets=["What a curious feeling, I must be shutting up like a telescope!"],
    output_snippet=json.dumps({"emotion": "curious", "intent": "exclamation"}),
)

# 2nd Sample
alice_line_classifier.add_sample(
    input_snippets=["Which way ought I to go from here?"],
    output_snippet=json.dumps({"emotion": "confused", "intent": "question"}),
)

# 3rd Sample
alice_line_classifier.add_sample(
    input_snippets=["Oh dear, I do hope this fall will ever come to an end."],
    output_snippet=json.dumps({"emotion": "afraid", "intent": "statement"}),
)

# 4th Sample
alice_line_classifier.add_sample(
    input_snippets=["How funny it seems to talk to a cat that keeps vanishing!"],
    output_snippet=json.dumps({"emotion": "amused", "intent": "exclamation"}),
)

# 5th Sample
alice_line_classifier.add_sample(
    input_snippets=["Are you quite certain that everyone here is mad?"],
    output_snippet=json.dumps({"emotion": "curious", "intent": "question"}),
)

# Add the Instruction to the Protocol
protocol.add_instruction(alice_line_classifier)

# Save the protocol. This produces the model.json that can be submitted to Databiomes for training.
protocol.save()
protocol.template()