'''
This is the word generator which we plan to use to generate the words to type.
The concept behind it is that it scales the probability of words appearing based
off of a value from the SAKT or alternatively it will eventually judge a number
of potential words based off of their probability after being passed through the
SAKT model.
'''

import transformers
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
import torch
import torch.nn.functional
from torch.distributions import Categorical
from collections import (defaultdict, Counter)
import string

# What words to focus on and their weights, for now this is generated
weighted_words = {
    'these': 0.147, 'runtime': 0.219, 'many': 0.684, 'they': 0.144, 'part': 0.769, 
    'it': 0.122, 'javascript': 0.619, 'are': 0.018, 'translucent': 0.366, 'write': 0.288, 
    'not': 0.958, 'into': 0.441, 'their': 0.891, 'languid': 0.585, 'other': 0.635, 
    'on': 0.553, 'effervescent': 0.382, 'conscientious': 0.742, 'ethernet': 0.262, 'be': 0.198, 
    'is': 0.881, 'surreptitious': 0.084, 'would': 0.161, 'called': 0.22, 'who': 0.483, 
    'some': 0.827, 'syzygy': 0.833, 'bureaucracy': 0.945, 'long': 0.022, 'output': 0.28, 
}

# Model setup

def get_causal_lm(model_name):
    model_name = "Qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # GPT-2 does not have an EOS token
    model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model

def get_qwen():
    return get_causal_lm("Qwen/Qwen1.5-1.8B")

def get_gpt2():
    return get_causal_lm("gpt2")

def get_flan_t5():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


class LogitsWordsBiaser(LogitsProcessor):
    def __init__(self, tokenizer, weighted_words, strength):
        self.strength = strength

        self.biases = {}
        for word in weighted_words:
            token_id = tokenizer.encode(word, add_special_tokens=False)[0]
            # Weigh all words equally for now
            self.biases[token_id] = self.strength

    def __call__(self, input_ids, scores):
        # Find greatest logit
        max_logit = torch.max(scores, dim=-1).values

        for token in self.biases:
            current_logit = scores[:, token]

            # Compute the gap between the logit and the maximum
            # so a bias of 1 will bring the logit equal to the maximum
            gap = (max_logit - current_logit) * self.biases[token]
            scores[:, token] += gap

        return scores


def generate_from_context(tokenizer, model, min_length, max_length, context, processors=None):
    inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False, truncation=True)
    outputs: transformers.GenerateOutput = model.generate(
        **inputs,
        min_new_tokens = min_length,
        max_new_tokens = max_length,
        return_dict_in_generate = True,
        do_sample=True,
        temperature=0.9,
        logits_processor=processors,
        top_p=0.95,
        output_scores = True,
    )

    return outputs



'''

Example 1:
Target Words: [keyboard, output, logic]
Sentence: The mechanical keyboard provides a tactile feel while the logic of the code determines the final output on screen.

Example 2:
Target Words: [network, database, cache]
Sentence: A slow network often requires a larger cache to ensure the database remains accessible during peak traffic hours.

Your Task:

'''

def get_prompt(weighted_words: dict[str, float]):
    words_to_include = ', '.join(weighted_words)
    
    context = f'''You are a professional typing instructor. Your goal is to write natural, coherent, and flowing sentences for a typing test. Use standard English grammar and keep punctuation limited to periods and commas. Avoid special characters or obscure vocabulary unless instructed otherwise.
Write a 10000 word text that naturally incorporates the target words.
Target Words: [{words_to_include}]
Output: "''' 
    return context


def generate_sentence_with_prompt(tokenizer, model, min_length, max_length, weighted_words, processors=None):
    context = get_prompt(weighted_words)
    output = generate_from_context(tokenizer, model, min_length, max_length, context, processors=processors)
    return tokenizer.batch_decode(output.sequences[0], skip_special_tokens=False)

def get_processors():
    temperature_warper = TemperatureLogitsWarper(1.5)
    biaser = LogitsWordsBiaser(tokenizer, weighted_words, 0.5)
    processors = LogitsProcessorList([temperature_warper, biaser])
    return processors

def generate_sentence_with_processors(tokenizer, model, min_length, max_length, weighted_words):
    processors = get_processors()
    return generate_sentence_with_prompt(tokenizer, model, min_length, max_length, weighted_words, processors=processors)


def count_target_words(weighted_words, sequence, defualt_count=0):
    text = [word.strip(string.punctuation).lower() for word in sequence.split()]

    counts = {}
    for word in weighted_words:
        counts[word] = defualt_count

    for token in text:
        if token in counts:
            counts[token] += 1
    
    total = 0
    coverage = 0
    for token in counts:
        total += counts[token]
        if counts[token] > 0:
            coverage += 1

    coverage /= len(counts)

    #print(f'Total usages: {total}, coverage: {coverage}')
    return {'total': total, 'coverage': coverage, }#'counts': counts}



if __name__ == '__main__':
    tokenizer, model = get_gpt2()
    min_length = 250
    max_length = 250

    runs = 3

    average_total = 0
    average_coverage = 0

    counts = []
    for i in range(runs):
        output = generate_sentence_with_processors(tokenizer, model, min_length, max_length, weighted_words)
        print(output)
        # Default count of -1 to ignore the words in the prompt
        count = count_target_words(weighted_words, output[0], -1)
        print(count)
        counts.append(count)
        average_total += count['total']
        average_coverage += count['coverage']

    average_total /= runs
    average_coverage /= runs

    print(counts)
    print(f'Average total: {average_total}, Average coverage: {average_coverage}')


