'''
Command to run this file:
python -m Unification.uni --csv final_draft\typing_log_v2.csv --bundle final_draft\artifacts\sakt_typing_bundle.pt
'''

# imports
from final_draft.SAKT_model_v2 import student_model

import numpy as np
import torch

class scoringModel(student_model):

    @torch.no_grad()
    def scoreWordsFromDataset(self, df, word2id, timeBinsMS, alpha=0.7):
        '''
        Extends student_model to score words based on how hard the user finds those words within the dataset.

        Returns formatted list of all words in the dataset with a difficulty score after.
        Difficulty (hardness) is a weighted mix of:
          - predicted probability of mistyping the next word
          - predicted time to type the next word (normalized)
        '''

        # Model in eval mode, and set up device to run it on (if CUDA is available, we use it)
        self.eval()
        device = next(self.parameters()).device

        # Variables..:
        wordScores = {}

        '''
        So, the way that this algorithm works is a little strange. Basically, i'm treating each user's test input
        as a complete "context" so-to-say. Bascially, for each user, we're going to look at all the words they type
        and try to assign a high score to what the most difficult word they have to type NEXT is, on average.

        So, we look at our sample dataset (around 2,000 samples) and look for small windows of difficulty. Perhaps
        a user finds the phrase "The Apple Is Red" difficult because the bigram [Apple, Is] is difficult for them.
        In that instance, we assign "is" a high score. 

        While this seems a strange methodology for right now, it becomes pertinent later when we think of wanting
        to construct a difficult exercise, rather than simply a difficult word. 

        If we only trained for difficult words, we would basically always try to guess the most complex word in the
        dataset. (e.g. we would always assign "Supercalifragilisticexpialidocious" a score of .999).

        What we want are challenging exercises, not necessarily challenging words.
        '''

        # Due to the architecture of the SAKT, we have a max step size within our embeddings, otherwise there
        # are errors created.
        # (This is due to the SAKT being a transformer with a maximum window size, but the context of our window
        # is N sized, with no maximum for N)
        # So we clamp down the max step size to an acceptable level.
        max_T = self.position_embedding.num_embeddings + 1
        step = max_T - 1

        # We can treat each 'user' as a compelte context..:
        for user_id, g in df.groupby('user_id'):

            # We have to convert the raw dataset to something useful to transform..:
            wordIDs = [word2id.get(w, 0) for w in g["word"].astype(str)]
            flags = [1 if int(m) == 0 else 0 for m in g["mistypes"]]
            times = g["time_ms"].astype(float).tolist()
            words_str = g["word"].astype(str).tolist()

            for start in range(0, len(wordIDs), step):

                # Take a chunk out of the dataset to examine..:
                w_chunk = wordIDs[start:start + max_T]
                f_chunk = flags[start:start + max_T]
                t_chunk = times[start:start + max_T]
                s_chunk = words_str[start:start + max_T]

                '''
                So, we have taken a chunk from the data and we're going to have a look at it. 
                That begs the question:
                How do we define difficulty??
                
                The approach I have taken is thus:
                I ask the SAKT model: Given your training, if the user just finised x word, can you predict how long
                they will take to type y word, and how many errors they will make while typing y word?
                Then, using those predictions, we can score all the possible next words (y-words) to then attempt
                to bias the LLM later.
                '''


                # So, we're going to do a bit of wrangling the data into nice, neat tensors
                # and organizing it a bit..:
                ques_in = torch.tensor(w_chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                ans_in = torch.tensor(f_chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                next_q = torch.tensor(w_chunk[1:], dtype=torch.long, device=device).unsqueeze(0)

                tb = np.digitize(np.array(t_chunk[:-1], dtype=np.float32), timeBinsMS)
                tb = np.clip(tb, 0, self.time_emb.num_embeddings - 1)
                tb_in = torch.tensor(tb, dtype=torch.long, device=device).unsqueeze(0)

                # Then ask the model for its predictions for all the words in the dataset
                # (You can see this kind of as a bigram type of system: x is the word and y is the word we
                # want to score in a bigram-like (x,y).)
                errLogits, predLogTime = self.forward(ques_in, ans_in, tb_in, next_q)

                # The LLM is designed to ingest probabilty-type bias (so .001 to .999) so we have to clamp &
                # normalize the outputs from the model.

                # Clamp (sigmoids - possibly an improvement to be made here later?)
                pCorrect = torch.sigmoid(errLogits)
                pMistype = 1.0 - pCorrect
                predTime = torch.expm1(predLogTime)

                # Normalize the data somewhat, we'll take out the 10th and 90th percentile outliers
                # (Usually produced by the tester getting up and grabbing a drink while the test ingest program is running)
                lo = torch.quantile(predTime, 0.1)
                hi = torch.quantile(predTime, 0.9)
                denom = torch.clamp(hi - lo, min=1e-6)
                t_norm = torch.clamp((predTime - lo) / denom, 0, 1)

                # Compute hardness score per next-word position:
                #   hardness = alpha * (mistype probability)
                #            + (1-alpha) * (normalized predicted time)
                # Alpha controls the tradeoff between how much of the score is controlled by the
                # number of mistakes vs how much of the score is controlled by the amount of time
                # a user takes to type a word.
                # I've used .7 for now, meaning that mistakes are more impactful than time
                # (from our dataset, time is pretty uniform around 1-ish seconds for all the words, so this
                # produces some more drastic swings in probability in general)
                hardness = alpha * pMistype + (1 - alpha) * t_norm
                hardness = hardness.squeeze(0).cpu().numpy()

                # We can collect up all the data into a list for the time being..:
                next_words = s_chunk[1:]
                for w, h in zip(next_words, hardness):
                    if w not in wordScores:
                        wordScores[w] = []
                    wordScores[w].append(float(h))

        # We want all the scores within (.000 -> .999), so we'll just round that off to 3 digits..:
        '''
        For Jeremy: We can probably grab this output here for use in a more dynamic system. If we did a
        SAKT Training pass prior to running this function, then grab the below output, it should be the same. 
        the only tricky part is that it may be a little tricky to be running and storing the model in real time.
        It's also possible that starting from a pretrained place and then getting more specific with the model
        could be a good approach.
        '''
        final_scores = [[w, round(float(np.mean(v)), 3)] for w, v in wordScores.items()]

        # For ease of reading, I have sorted the numbers highest to lowest in the CSV.
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores

# TESTING SYSTEMS. NOT NECESSARILY FINAL USE PRODUCT.
# This is just CLI crap. do what you want.
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os

    # The CSV output from the typing program has no headers. Fix that.
    def load_typing_csv(path):
        df = pd.read_csv(
            path,
            header=None,
            names=["user_id", "word", "time_ms", "mistypes"]
        )
        df["timestamp"] = df.groupby("user_id").cumcount()
        return df

    # Vocabulary buildinger
    def build_word_vocab(df, min_freq=1):
        counts = df["word"].value_counts()
        words = counts[counts >= min_freq].index.tolist()
        word2id = {w: (i + 1) for i, w in enumerate(words)}
        return word2id

    # bucket factory
    def make_time_bins(df, buckets=10, lo_pct=1, hi_pct=99):
        times = df["time_ms"].astype(float).to_numpy()
        if len(times) == 0:
            return np.array([], dtype=np.float32)

        lo = np.percentile(times, lo_pct)
        hi = np.percentile(times, hi_pct)

        lo = max(lo, 1.0)
        hi = max(hi, lo + 1.0)

        edges = np.geomspace(lo, hi, buckets + 1)[1:-1]
        edges = np.unique(edges.astype(np.float32))
        return edges

    # CL args and shit
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to typing log CSV (no header): user_id,word,time_ms,mistypes")
    ap.add_argument("--bundle", required=True, help="Path to trained bundle .pt file")
    ap.add_argument("--out", default="word_scores.csv", help="Where to save word scores CSV")
    ap.add_argument("--alpha", type=float, default=0.7, help="1.0 only mistype, 0.0 only time (default 0.7)")
    ap.add_argument("--top", type=int, default=25, help="Print top N hardest words")
    ap.add_argument("--min_freq", type=int, default=1, help="Min word frequency for vocab if bundle lacks word2id")
    args = ap.parse_args()

    # CUDA stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Grab the dataset
    df = load_typing_csv(args.csv)
    if len(df) == 0:
        raise SystemExit("CSV is empty: " + args.csv)

    # Grab artifacts and do bundle stuff
    bundle = torch.load(args.bundle, map_location=device)
    opt_dict = bundle.get("opt", {})

    word2id = bundle.get("word2id", None)
    if word2id is None:
        word2id = build_word_vocab(df, min_freq=args.min_freq)

    timeBinsMS = bundle.get("time_bins_ms", None)
    if timeBinsMS is None:
        timeBinsMS = make_time_bins(df, buckets=10, lo_pct=1, hi_pct=99)
    else:
        timeBinsMS = np.array(timeBinsMS, dtype=np.float32)

    # Hyperparams
    state_size = opt_dict.get("state_size", 200)
    num_heads = opt_dict.get("num_heads", 5)
    dropout = opt_dict.get("dropout", 0.1)
    num_skills = max(word2id.values()) + 1

    # Model init
    model = scoringModel(
        num_skills=num_skills,
        state_size=state_size,
        num_heads=num_heads,
        dropout=dropout,
        infer=True
    ).to(device)

    # Load weights, set eval mode (also do this in the file so a bit redundant)
    if "state_dict" not in bundle:
        raise SystemExit("Bundle missing 'state_dict'. Did you save the bundle correctly?")
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    # Initial scoring of all the words. Note that this is also useful for evaluating a user's performance in a
    # specific exercise.
    scores = model.scoreWordsFromDataset(df, word2id, timeBinsMS, alpha=args.alpha)

    # Ease of use. See hardest words in the console (this was very useful during testing)
    print(f"\nTop {min(args.top, len(scores))} hardest words:")
    for w, s in scores[:args.top]:
        print(f"{w:20s} {s:.3f}")

    # Save to CSV, to be read later on by the LLM.
    out_df = pd.DataFrame(scores, columns=["word", "score"])
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} word scores to: {args.out}")