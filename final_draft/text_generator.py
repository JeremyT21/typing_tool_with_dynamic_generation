"""
This is the word generator which we plan to use to generate the words to type.
The concept behind it is that it scales the probability of words appearing based
off of a value from the SAKT or alternatively it will eventually judge a number
of potential words based off of their probability after being passed through the
SAKT model.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper
)

import torch


def get_gpt2():
    # S-M: Rewrote this function and removed other references.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model

class LogitsWordsBiaser(LogitsProcessor):

    def __init__(self, tokenizer, weighted_words, strength):

        self.biases = {}
        self.strength = strength

        for word, weight in weighted_words.items():

            tokens = tokenizer.encode(word, add_special_tokens=False)

            if len(tokens) == 0:
                continue

            token_id = tokens[0]

            self.biases[token_id] = float(weight)


    def __call__(self, input_ids, scores):

        for token_id, weight in self.biases.items():

            scores[:, token_id] += weight * self.strength

        return scores

def get_prompt(weighted_words):

    # S-M: Amended somewhat; works essentially the same.

    words = ", ".join(list(weighted_words.keys())[:20])

    return f"""
You are generating text for a typing practice program.

Write ONE natural English sentence between 15 and 25 words.

Try to include some of these words if possible:
{words}

Rules:
- Only letters, spaces, commas, and periods
- No explanations
- No lists
- No quotes

Sentence:
"""

class TextGenerator:

    # S-M: Amended heavily.
    def __init__(self, weighted_words=None, bias=2.0):

        if weighted_words is None:
            weighted_words = {}

        # Using GPT-2
        self.tokenizer, self.model = get_gpt2()

        # Storing our weights and biases within the object.
        self.weighted_words = weighted_words
        self.bias = bias


    def set_weighted_words(self, weighted_words):
        # Setter
        self.weighted_words = weighted_words


    def generate_sentence(self, min_length=30, max_length=60):

        # Goes about actually generating a sentence.
        prompt = get_prompt(self.weighted_words)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        processors = LogitsProcessorList([
            TemperatureLogitsWarper(1.2),
            LogitsWordsBiaser(self.tokenizer, self.weighted_words, self.bias)
        ])

        outputs = self.model.generate(
            **inputs,
            min_new_tokens=min_length,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.95,
            logits_processor=processors
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        sentence = text.split("Sentence:")[-1].strip()

        sentence = sentence.split("\n")[0].strip()

        sentence = sentence.replace('"', "")

        # Prevent extremely short outputs
        # Was having some issues with this (i.e. single words)
        # When the code runs, you can see this rejecting runs when it gets ones
        # which are too short; it says something about adding padding whenever
        # you make a call to the GPT-2.
        if len(sentence.split()) < 10:
            return self.generate_sentence()

        return sentence