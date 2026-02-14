'''
This is the word generator which we plan to use to generate the words to type.
The concept behind it is that it scales the probability of words appearing based
off of a value from the SAKT or alternatively it will eventually judge a number
of potential words based off of their probability after being passed through the
SAKT model.

The current version doesn't work too well because most of the time its targeted
words have a probability of 0 meaning multiplying their probability doesn't do
anything. We try adding to their probability instead but that leads to the target
words appearing in a context in which they don't make any sense.
'''

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
import torch
import torch.nn.functional
from torch.distributions import Categorical

# What words to focus on and their weights
weighted_words = {
    "food": 1.5,
    "dog": 1.5,
}

# How many tokens to generate
words_to_generate = 100

# Sampling temperature
temperature = 20.0

# Whether to add to the probability of the output words appearing. This
# can lead them to appear in a context in which they don't make sense.
should_add_scale = True

# Factor to scale the words values by when multiplying their probability distribution
scale_factor = 2

# Initial starting prompt to the LM. Should generate varied text, good for typing.
# Here it is attempting to increase the likelyhood that the target words probabilities
# will not be 0

context = "I like dogs and I like food "


# Model setup

#model_name = "google/flan-t5-base"
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 does not have an EOS token
model.config.pad_token_id = model.config.eos_token_id


# Tokenize the weighted words for quick lookup

tokenized_weighted_words = {}
for word in weighted_words:
    token_id = tokenizer(word, return_tensors="pt").input_ids[0]
    tokenized_weighted_words[token_id] = weighted_words[word]


# The final text after running the model excluding the initial prompt
final_text = ""

for i in range(words_to_generate):
    inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False, truncation=True)
    outputs = model.generate(
        **inputs,
        min_new_tokens = 1,
        max_new_tokens = 1,
        return_dict_in_generate = True,
        output_scores = True,
    )
    
    # Since we want to modify the probability distributions, we need the logits directly
    logits = outputs.scores[0]
    logits /= temperature

    # Convert to a probability distribution
    softmax = torch.nn.functional.softmax(logits, dim=1)[0]

    # Apply the weighting
    for token in tokenized_weighted_words:
        # print(f'token: {token}, prob: {softmax[token]}, logit: {logits[0][token]}')
        softmax[token] *= tokenized_weighted_words[token] * scale_factor
        if should_add_scale:
            softmax[token] += tokenized_weighted_words[token] / 100.0
    
    # At this point 'softmax' will not technically contain a probability distribution

    #softmax = torch.nn.functional.softmax(softmax, dim=1)

    # Sample from the new distribution
    categorical = Categorical(softmax)
    next_token = categorical.sample().item()
    # next_token = torch.argmax(softmax, dim=1)

    token_text = tokenizer.decode(next_token, skip_special_tokens=True)
    
    # Add the new token to the context so it will be fed back into the LM
    context += token_text
    final_text += token_text


print(context)

