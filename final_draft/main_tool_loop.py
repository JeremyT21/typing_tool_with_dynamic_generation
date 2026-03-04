import time
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

import importlib.util as imput #used for module creation / loading

import Levenshtein #used for similarity between 2 words

#below function imports other .py files and exports them as modules,
#so the other file py code can be ran from this file
def import_and_execute_py_file(path_to_file, name_for_module):
    path_to_file = os.path.abspath(path_to_file)

    #sets up the imported py files configuration
    imported_file_spec = imput.spec_from_file_location(name_for_module, path_to_file)

    #creates the module so the code can be ran
    created_module = imput.module_from_spec(imported_file_spec)
    sys.modules[name_for_module] = created_module

    #runs the code
    imported_file_spec.loader.exec_module(created_module)
    
    return created_module

#loads pickable words given path of the txt file containing them
def load_words_that_can_be_picked(path):
    words = []
    with open(path, "r") as file:
        for line in file:
            current_word = line.strip().lower()
            current_word = re.sub(r"[^a-z0-9]+","",current_word)
            if current_word not in words:
                words.append(current_word)
    return words

#below function uses levenshtein to calculate word similarities between words, given a list of words
#it will return similarites for the 10 MOST SIMILAR TO TARGET WORD words by default
def similarities_for_words(target_word, words, returned_words_num = 10):
    scored_words = []

    #below loop goes through words, skipping the target_word from comparing with itself
    for word in words:
        if word == target_word:
            continue
        
        similarity_dist = Levenshtein.distance(target_word, word)
        difference_of_length = abs(len(target_word) - len(word))
        scored_words.append((similarity_dist, difference_of_length, word))

    #now sorts words by values    
    scored_words.sort()

    #now extract only certain amount of most similar words
    most_similar_words = []
    for i in scored_words[:returned_words_num]:
        word = i[2]
        most_similar_words.append(word)

    return most_similar_words

#below function ensures user types a sentence out word by word,
#and that the pickable words are the only words that can have data recorded and appended
def show_sentence_one_word_per(sentence, words_can_be_picked):
    current_word_data = []

    print("\nType each word, once you are finished typing a word press 'Enter' (quit with ctrl c)\n")
    
    for word in sentence.split():
        current_type_target = word.strip().lower()
        current_type_target = re.sub(r"[^a-z0-9]+","",current_type_target)
        print(f"|| Currently typing word: '{current_type_target}' ||")

        time0 = time.time()
        mistypes = 0

        while True:
            text_entered = input("|| >>> ")

            text_entered = text_entered.strip().lower()
            text_entered = re.sub(r"[^a-z0-9]+","",text_entered)

            if text_entered == current_type_target:
                print("  CORRECT :)\n")
                break
            else:
                mistypes += 1
                print("   WRONG :(\n")

        time1 = time.time()#time after input
        time_in_ms = (time1 - time0) * 1000

        #only words seen by SAKT will have data recorded
        if current_type_target in words_can_be_picked:
            current_word_data.append((current_type_target,time_in_ms,mistypes))
    
    return current_word_data

#loading the SAKT artifact
def load_sakt_bundle(path, device):
    bundle = torch.load(path, map_location=device)
    word2id = bundle.get("word2id")
    time_bins_ms = bundle.get("time_bins_ms")
    time_bins_ms = np.array(time_bins_ms)
    opt = bundle.get("opt", {})

    return bundle, word2id, time_bins_ms, opt

if __name__ == "__main__":
    bundle_path = "./artifacts/sakt_typing_bundle.pt"
    pickable_path = "./pickable_words.txt"
    scoring_path = "./unification/uni.py"#difficulty scores
    lm_path = "./text_generator.py"
    log_csv = "./datasets/largest_dataset_2.csv"

    user_id = "u1"
    alpha = 0.7
    top_difficult_words = 1 #only want sentences to encorporate top most difficult word
    sentence_try_generating = 4 #generate max of 4 sentences until best is generated
    similar_words = 10
    min_sentence_len = 14
    max_sentence_len = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pickable_words = load_words_that_can_be_picked(pickable_path)

    scorer_module = import_and_execute_py_file(scoring_path, "user_scorer_module")
    lm_module = import_and_execute_py_file(lm_path, "user_LM_module")

    #HERE IS WHERE FUNCTIONS ARE CALLED FROM GROUP MEMBER CREATED .py FILES:
    scoring_model = scorer_module.scoringModel
    get_gpt2 = lm_module.get_gpt2
    generate_sentence = lm_module.generate_sentence_with_prompt

    #load the bundle
    bundle, word2id, time_bins_ms, opt = load_sakt_bundle(bundle_path, device)

    #defining the scoring model using parameters from artifact SAKT .pt
    state_size = opt.get("state_size")
    dropout = opt.get("dropout")
    num_heads = opt.get("num_heads")
    num_skills = max(word2id.values())+1

    model = scoring_model(
        num_skills=num_skills,
        state_size=state_size,
        num_heads=num_heads,
        dropout=dropout,
        infer=True           
    ).to(device)

    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    #load the language model
    tokenizer, language_model = get_gpt2()
    language_model.to(device)
    language_model.eval()

    #for storing data of current word
    current_word_data = pd.DataFrame(columns=["user_id","word","time_ms","mistypes"])
    sentence = "This is starting sentence for tool!"

    print("\n|| Personalized Typing Tool ||\n")

    #main loop typing:
    while True:
        print(sentence)
        results = show_sentence_one_word_per(sentence, pickable_words)
        
        #if a word is not generated or null space is accidentally shown it will be skipped
        if not results:
            continue

        #adding the current words data to the csv
        column_start = len(current_word_data)#needs to go through all indexes of data
        for current_index, (word, time_ms, mistypes) in enumerate(results):
            current_word_data.loc[column_start + current_index, :] = {
                "user_id": "u1",
                "word": word,
                "time_ms": time_ms,
                "mistypes": mistypes
            }

        added_word_data = current_word_data.iloc[column_start : column_start + len(results)][["user_id","word","time_ms","mistypes"]]
        added_word_data.to_csv(log_csv, mode="a", header=False, index=False)

        #getting scores on only a recent history, last 200 words for example
        num_word_history = 200
        recent_scores = current_word_data.tail(num_word_history).copy()#need to copy
        
        #calling uni score function
        scores = model.scoreWordsFromDataset(recent_scores, word2id, time_bins_ms, alpha=alpha)

        #ensures only pickable words scores are used
        #by building list of scores for pickable words
        scores_needed = []
        for word,score in scores:
            if word in pickable_words:
                scores_needed.append((word, score))

        scores = scores_needed

        #in the current case only want top 1 most difficult word
        hard_words = []
        for hardest_words_and_scores in scores[:top_difficult_words]:
            hard_words.append(hardest_words_and_scores[0])

        print(f"for testing: hard word: {hard_words[0]}")
        
        #for now we are also looking through pickable words
        #to generate a sentence with a similar pickable word

        words_to_focus_on = []
        for current_hard_word in hard_words:
            words_to_focus_on.append(current_hard_word)
            words_to_focus_on.extend(similarities_for_words(current_hard_word, pickable_words))#need to extend since multiple words returned by function

        #FUTURE IMPROVEMENT: just using a simple way of making certain word difficulties higher
        #right now first word difficulty is 1, and the rest in the hard words to focus on are 0.5
        word_weights = {}
        for index, current_word in enumerate(words_to_focus_on):
            if index == 0:
                word_weights[current_word] = 1.0
            else:
                word_weights[current_word] = 0.5

        #generating the sentence
        #for now no best sentence keeping is used, just the first sentence created 
        best_sentence = None
        best_sentences_num_created = -1

        #for current_generated_sentence_num in range(sentence_try_generating):
        generated_text = generate_sentence(
            tokenizer,
            language_model,
            min_length=min_sentence_len,
            max_length=max_sentence_len,
            weighted_words=word_weights
        )

        #waiting to see how group member is coding sentence return
        if isinstance(generated_text, list):
            returned_text = generated_text[0]
            sentence = returned_text
        else:
            sentence = str(generated_text)
        
        print("\nPlease type next sentence:\n")
        print(sentence)