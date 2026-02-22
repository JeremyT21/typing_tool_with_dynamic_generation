# Dynamically Generated Typist Trainer
### Jeremy Thummel, Andrew Hunter, & Sam McAdoo

## Goals & Background

The goal of this project is to replicate and implement this [Paper](https://aclanthology.org/2023.acl-long.567/), written by Peng Cui & Mrinmaya Sachan. The goal of this paper, and our project, is to create a program which can dynamically generate tailored typing exercises for a user. That is, an algorithm which learns which words a user finds difficult and constructs novel exercises (new phrases and sentences) which contain those words.

In essence, this project contains two distinct algorithms: A SAKT (Self Attentive Knowledge Training) model, and a more typical language model which ingests the results found by the SAKT to produce new exercises. The two models work together to enable the tool, in a manner similar to a GAN, with the tool's user taking the place of the discerner algorithm. This archetecture is the one suggested in the paper, and in our findings so far works well.

## Implementation & Running

You can view the current implementations of the two algorithms in this repo, under "final". By running the Python script found there, you can begin using the tool and allowing it to learn from your typing. The weights and biases of the tool are pre-stored, to save computation time, so all you should need is a working install of Python 3.12 on your system. You will note additionally that, currently, the tool stores its' "findings" locally, and you may find this information interesting: The tool scores how much difficulty with a given word as a numeric result of the SAKT algorithm.

## To-Do & Further Steps
