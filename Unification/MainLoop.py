# Imports (many)
import pandas as pd
import torch
import numpy as np

from Unification.TypingGame import TypingGame
from final_draft.text_generator import TextGenerator
from Unification.uni import scoringModel
from pathlib import Path

'''
This file handles all the logic for the main event, as it were. Spins up both algorithms,
then spins up the game so we can get 'crackin.
'''

# We load the base SAKT training, done on a handmade datatset of one of the group
# members' typing.

def load_sakt(bundle_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(bundle_path, map_location=device)

    opt = bundle["opt"]
    word2id = bundle["word2id"]
    time_bins = np.array(bundle["time_bins_ms"], dtype=np.float32)

    model = scoringModel(
        num_skills=max(word2id.values()) + 1,
        state_size=opt["state_size"],
        num_heads=opt["num_heads"],
        dropout=opt["dropout"],
        infer=True
    ).to(device)

    model.load_state_dict(bundle["state_dict"])
    model.eval()

    return model, word2id, time_bins

# Load up the artifact
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

bundle_path = PROJECT_ROOT / "final_draft" / "artifacts" / "sakt_typing_bundle.pt"

print("Loading SAKT bundle from:", bundle_path)

sakt_model, word2id, time_bins = load_sakt(bundle_path)


# The generator takes slightly different weights than we get from the SAKT,
# so this gets those worked out.
def scores_to_weights(scores, top_k=30):

    weights = {}

    for word, score in scores[:top_k]:
        weights[word] = float(score)

    return weights


# Initialize all the stuff!!

# Text generator (GPT-2 in this case)
generator = TextGenerator()

# need to keep a log of all the typing info to feed to the SAKT.
typing_log = pd.DataFrame(columns=["user_id", "word", "time_ms", "mistypes"])

# Placeholder;
user_id = 1
# # of current sentence
sentence_counter = 0

# Initialize game window
game = TypingGame()



while True:

    # Track what sentence we're on..:
    sentence_counter += 1

    # All these prints are mostly debug
    print(f"Sentence: {sentence_counter}")

    # Grab a sentence from the generator..:
    sentence = generator.generate_sentence()

    print("Generated sentence:")
    print(sentence)

    # Update the game interface with our sentence..:
    result_df = game.run(sentence)

    # Grab results from the game..:
    result_df["user_id"] = user_id
    typing_log = pd.concat([typing_log, result_df], ignore_index=True)

    print("Updating difficulty model...")

    # Grab scores from the previous work from the SAKT.
    scores = sakt_model.scoreWordsFromDataset(
        typing_log,
        word2id,
        time_bins
    )

    # Transform those scores according to the weights above..:
    weighted_words = scores_to_weights(scores)

    # Input those into the generator..:
    generator.set_weighted_words(weighted_words)


    # More debug
    print("\nTop difficult words:")
    for word, score in scores[:10]:
        print(f"{word:15s} {score:.3f}")