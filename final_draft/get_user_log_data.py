import csv
import os
import random
import time

#need to install random_word library (use pip install random_word)
#from random_word import RandomWords

#now using wordfreq to get frequently used words, not obscure words from RandomWords
from wordfreq import top_n_list

#need to filter out commonly used bad words
from better_profanity import profanity

csv_path = "typing_log_v2.csv"
user_id = "u1"

header = ["user_id", "word", "time_ms", "mistypes"]

'''
#old word generation function:
def generate_word_list(n):
    r = RandomWords()
    words = []

    while len(words) < n:
        words.append(r.get_random_word().lower())

    return words
    
words_to_type = generate_word_list(520)
'''

def generate_word_list(n):
    #https://github.com/rspeer/wordfreq used to get frequently used english words
    common_words = top_n_list("en", n)#"en" ensures english
    
    #numbers can still be returned since they are commonly typed, to remove referenced: https://stackoverflow.com/questions/3159155/how-to-remove-all-integer-values-from-a-list-in-python
    common_words_with_no_numbers = [word for word in common_words if not (word.isdigit() or word[0] == "-" and word[1:].isdigit())]

    # Filter all single characters (such as 'd', 'r', 't')..:
    common_words_with_no_numbers = [word for word in common_words_with_no_numbers if not (len(word) == 1)]

    #filter out all bad words - used https://pypi.org/project/better-profanity/    
    no_bad_words = [word for word in common_words_with_no_numbers if not profanity.contains_profanity(word)]

    #this will be used to show the language model which words have difficulties generated from the SAKT
    #(since these words are what the SAKT will use to train on)
    with open("pickable_words.txt", "w") as file:
        for word in no_bad_words:
            file.write(f"{word}\n")

    return no_bad_words

words_to_type = generate_word_list(500)

#adding rows to the csv of inputted data
def append_row(path, user_id, word, time_ms, mistypes):
    with open(path, "a", newline="", encoding="utf-8") as writing_to_csv:
        csv.writer(writing_to_csv).writerow([user_id, word, time_ms, mistypes])

#format words
def reformatted(word):
    return word.strip().lower()

def get_word_type_data():
    while True:
        target = random.choice(words_to_type)
        target_norm = reformatted(target)

        print(f"please type word: {target}")
        start = time.perf_counter()
        wrong_enters = 0

        while True:
            typed = input("TYPE: ")
            if reformatted(typed) == "quit now":
                print("all words typed")
                return

            if reformatted(typed) == target_norm:
                #gets time until enter pressed after typing word in terminal
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                append_row(csv_path, user_id, target, elapsed_ms, wrong_enters)
                print(f"RIGHT ANSWER - ADDED TO CSV: {user_id},{target},{elapsed_ms},{wrong_enters}\n")
                break
            else:
                wrong_enters += 1
                print("WRONG - RETYPE")

#run the logging loop
get_word_type_data()