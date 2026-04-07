"""
This is the word generator which we plan to use to generate the words to type.
The concept behind it is that it scales the probability of words appearing based
off of a value from the SAKT or alternatively it will eventually judge a number
of potential words based off of their probability after being passed through the
SAKT model.
"""

from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
import torch
import time
from collections import (defaultdict, Counter)
import string


def get_causal_lm(model_name):
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
    def __init__(self, weighted_words=None, bias=2.0, model = None, tokenizer = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if weighted_words is None:
            weighted_words = {}

        if model is None or tokenizer is None:
            # Using GPT-2
            self.tokenizer, self.model = get_qwen()
        else:
            self.tokenizer = tokenizer
            self.model = model
        
        self.model = self.model.to(self.device)

        # Storing our weights and biases within the object.
        self.weighted_words = weighted_words
        self.bias = bias


    def set_weighted_words(self, weighted_words):
        # Setter
        self.weighted_words = weighted_words


    def generate_sentence(self, min_length=30, max_length=60):

        # Goes about actually generating a sentence.
        prompt = get_prompt(self.weighted_words)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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

def benchmark(tokenizer_model, weighted_words, model_name):
    print('Benchmarking', model_name)

    # Needs to be updated to work with the current version
    tokenizer, model = tokenizer_model
    generator = TextGenerator(weighted_words, model = model, tokenizer = tokenizer)

    min_length = 250
    max_length = 250

    runs = 50

    average_total = 0
    average_coverage = 0
    average_time = 0

    counts = []
    for i in range(runs):
        start = time.time()
        output = generator.generate_sentence()
        run_time = time.time() - start
        print(f'Time: {run_time}')
        print(output)
        count = count_target_words(weighted_words, output)
        print(count)
        counts.append(count)
        average_total += count['total']
        average_coverage += count['coverage']
        average_time += run_time

    average_total /= runs
    average_coverage /= runs
    average_time /= runs

    print(counts)
    results = f'Model: {model_name}, Runs: {runs}, Average total: {average_total}, Average coverage: {average_coverage}, Average time {average_time}'
    print(results)
    return results


# If this file is run on its own, run benchmarks
if __name__ == '__main__':
    sample_weighted_words = {
        'these': 0.147, 'runtime': 0.219, 'many': 0.684, 'they': 0.144, 'part': 0.769, 
        'it': 0.122, 'javascript': 0.619, 'are': 0.018, 'translucent': 0.366, 'write': 0.288, 
        'not': 0.958, 'into': 0.441, 'their': 0.891, 'languid': 0.585, 'other': 0.635, 
        'on': 0.553, 'effervescent': 0.382, 'conscientious': 0.742, 'ethernet': 0.262, 'be': 0.198, 
        'is': 0.881, 'surreptitious': 0.084, 'would': 0.161, 'called': 0.22, 'who': 0.483, 
        'some': 0.827, 'syzygy': 0.833, 'bureaucracy': 0.945, 'long': 0.022, 'output': 0.28, 
    }

    results = []
    results.append(benchmark(get_flan_t5(), sample_weighted_words, 'Flan-T5-Base'))
    results.append(benchmark(get_gpt2(), sample_weighted_words, 'GPT2'))
    results.append(benchmark(get_qwen(), sample_weighted_words, 'Qwen'))

    print('\n'.join(results))
