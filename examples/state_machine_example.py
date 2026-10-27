import model_train_protocol as mtp

# Cheshire Cat Question Router

# NOTE: For most state machines, the CSV interface is the recommended approach. It only requires Input, Output, and
# Reference columns and does not require any code. Use this Python approach only when you need custom Tokens,
# Token descriptions, or multiple input lines. See https://docs.databiomes.com/training/csv/overview/

# This example protocol demonstrates a StateMachineInstruction. Instead of generating a free-form reply, the model
# reads a single line spoken by Alice (from "Alice's Adventures in Wonderland") and routes it to one of a fixed list
# of states. Your application then decides what to do with the state.
#
# A state machine protocol contains exactly one StateMachineInstruction, does not use final tokens, and does not
# allow numeric output.

protocol = mtp.Protocol(name="state_machine_example", inputs=1, encrypt=False)

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
    "The Cat does not reply with words. Instead, it sorts each thing Alice says into the kind of question she is asking.")
protocol.add_context(
    "Alice asks about directions, about who the Cat is, about madness, and about where she has ended up.")
protocol.add_context(
    "Sometimes she is not asking a question at all, and simply says goodbye before wandering off again.")
protocol.add_context(
    "Each line Alice speaks is routed to exactly one state, and never to more than one.")
protocol.add_context(
    "The states are used by Wonderland to decide which part of the story should respond to Alice next.")

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

# -------------------- State Machine Instruction --------------------

# The states are the complete list of tokens the model can respond with. The model responds with a single state token
# in place of a generated string.
token_direction: mtp.Token = mtp.Token("Direction", desc="Alice asks which way to go.")
token_identity: mtp.Token = mtp.Token("Identity", desc="Alice asks who or what the Cat is.")
token_madness: mtp.Token = mtp.Token("Madness", desc="Alice asks about madness.")
token_location: mtp.Token = mtp.Token("Location", desc="Alice asks where she is.")
token_farewell: mtp.Token = mtp.Token("Farewell", desc="Alice says goodbye.")
states = [token_direction, token_identity, token_madness, token_location, token_farewell]

# Construct the Input format. A state machine accepts one or more input TokenSets; here we use a single line.
alice_line_input: mtp.StateMachineInput = mtp.StateMachineInput(
    tokensets=[tree_english_alice_talk],
)

# Create the Instruction. The output is derived automatically from the state tokens,
# so we only need to provide the input and the states.
alice_line_router: mtp.StateMachineInstruction = mtp.StateMachineInstruction(
    input=alice_line_input,
    states=states,
)

# Give the Instruction some background context.
alice_line_router.add_context(
    "The Cheshire Cat routes each line Alice speaks to the kind of question it represents.")
alice_line_router.add_context(
    "Every response is exactly one of the defined state tokens.")

# Add at least 10 samples, with at least 3 for each state token. Each sample maps an input line to one state token.

# DIRECTION
alice_line_router.add_sample(
    input_snippets=["Which way ought I to go from here?"],
    state=token_direction,
)
alice_line_router.add_sample(
    input_snippets=["Can you tell me which road leads out of this wood?"],
    state=token_direction,
)
alice_line_router.add_sample(
    input_snippets=["Should I follow the path to the left or to the right?"],
    state=token_direction,
)

# IDENTITY
alice_line_router.add_sample(
    input_snippets=["Please, would you tell me what sort of creature you are?"],
    state=token_identity,
)
alice_line_router.add_sample(
    input_snippets=["Are you a cat, or something else entirely?"],
    state=token_identity,
)
alice_line_router.add_sample(
    input_snippets=["Who are you, and why do you keep grinning so?"],
    state=token_identity,
)

# MADNESS
alice_line_router.add_sample(
    input_snippets=["Are you quite certain that everyone here is mad?"],
    state=token_madness,
)
alice_line_router.add_sample(
    input_snippets=["How do you know that I am mad as well?"],
    state=token_madness,
)
alice_line_router.add_sample(
    input_snippets=["Is there anyone in this place who is not mad?"],
    state=token_madness,
)

# LOCATION
alice_line_router.add_sample(
    input_snippets=["Where in the world have I landed?"],
    state=token_location,
)
alice_line_router.add_sample(
    input_snippets=["What is this strange place called?"],
    state=token_location,
)
alice_line_router.add_sample(
    input_snippets=["Am I still in England, or somewhere else altogether?"],
    state=token_location,
)

# FAREWELL
alice_line_router.add_sample(
    input_snippets=["Goodbye then, I shall find the way myself."],
    state=token_farewell,
)
alice_line_router.add_sample(
    input_snippets=["I must be going now, thank you all the same."],
    state=token_farewell,
)
alice_line_router.add_sample(
    input_snippets=["Farewell, Cat, I hope we meet again someday."],
    state=token_farewell,
)

# -------------------- Guardrail --------------------

# Guardrails catch inputs that are irrelevant to the model's domain.
guardrail_english: mtp.Guardrail = mtp.Guardrail(
    good_prompt="A question or remark Alice speaks aloud in Wonderland with 1-20 words",
    bad_prompt="A prompt that is irrelevant and off topic with 1-20 words",
    bad_output="GUARDRAIL",
)

# Add at least 3 samples of bad prompts to the Guardrail.
guardrail_english.add_sample("explain quantum mechanics.")
guardrail_english.add_sample("who will win the next american election?")
guardrail_english.add_sample("what is the capital of Spain?")

# Apply the Guardrail to the first TokenSet in the Instruction input.
alice_line_router.add_guardrail(guardrail=guardrail_english, tokenset_index=0)

# Add the Instruction to the Protocol
protocol.add_instruction(alice_line_router)

# Save the protocol. This produces the model.json that can be submitted to Databiomes for training.
protocol.save()
protocol.template()
