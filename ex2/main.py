# "Code is poetry"

import nltk
import numpy as np
from collections import defaultdict
from copy import deepcopy
import re

# Use !* because apparently, the word "not" is tagged with * 
START_MARKER = '!*'
STOP_MARKER = 'STOP'


def prep_part_of_speech(target):
    if '+' in target or '-' in target and len(target) > 2 and target != '--':
        matches = re.search(r'(.+?)([\-\+])', target)
        return matches.group(1)
    return target


def prep_dataset(dataset):
    return list(map(lambda sentence: list(map(lambda pair: (pair[0], prep_part_of_speech(pair[1])), sentence)), dataset))

def transform_pseudo(dataset):
    words_histogram = defaultdict(int)
    for sentence in dataset:
        for word, tag in sentence:
            words_histogram[word] += 1
    low_frequency_words = [word for word, count in words_histogram.items() if count <= 4]
    for sentence in dataset:
        for i in range(len(sentence)):
            word = sentence[i][0]
            tag = sentence[i][1]
            if word in low_frequency_words:
                if word.isdigit():
                    if len(word) == 2:
                        pseudo = "2Digits"
                    else:
                        pseudo = "allDigits"
                elif word.isupper():
                    pseudo = "allCaps"
                elif word[0].isupper() and word[-1] == ".":
                    pseudo = "capPeriod"
                elif word[0].isupper():
                    pseudo = "initCaps"
                else:
                    pseudo = word
                sentence[i] = (pseudo, tag)
    return dataset

def create_words_map_tags_histogram(dataset):
    words_map = defaultdict(lambda: defaultdict(int))
    tags_histogram = defaultdict(int)
    for sentence in dataset:
        for word, part_of_speech in sentence:
            words_map[word][part_of_speech] += 1
            tags_histogram[part_of_speech] += 1
    tags_histogram[START_MARKER] = len(training_data)
    tags_histogram[STOP_MARKER] = len(training_data)
    return words_map, tags_histogram



# nltk.download("brown")
tagged_sents = list(nltk.corpus.brown.tagged_sents(categories="news"))
training_size = int(len(tagged_sents) * 0.9)
training_data = prep_dataset(tagged_sents[:training_size])
test_data = prep_dataset(tagged_sents[training_size:])
words_map, tags_histogram = create_words_map_tags_histogram(training_data)


# Baseline - most likely tag with no other assumptions
most_probable_tag = dict()
for word in words_map:
    most_probable_tag[word] = max(words_map[word].items(), key=lambda pair: pair[1])[0]

known_words_count = 0
known_hits_count = 0
unknown_words_count = 0
unknown_hits_count = 0

for sentence in test_data:
    for word, real_tag in sentence:
        if word in most_probable_tag:
            predicted_tag = most_probable_tag[word]
            known_words_count += 1
            if predicted_tag == real_tag:
                known_hits_count += 1
        else:
            predicted_tag = "NN"
            unknown_words_count += 1
            if predicted_tag == real_tag: 
                unknown_hits_count += 1
known_words_accuracy = known_hits_count / known_words_count
unknown_words_accuracy = unknown_hits_count / unknown_words_count
total_accuracy = (known_hits_count + unknown_hits_count) / (known_words_count + unknown_words_count)

print("Known words prediction err:", 1 - known_words_accuracy)
print("Unknown words prediction err:", 1 - unknown_words_accuracy)
print("Total prediction err:", 1 - total_accuracy)

# Bigram HMM
def viterbi(sentence, transitions, emissions):
    n = len(sentence)
    
    # Calling pi pie so I won't confuse it with PI
    pie = [defaultdict(float) for i in range(n + 1)]
    tags = set(tags_histogram.keys())
    for tag in tags:
        pie[0][tag] = (None, 1.0)
    for k in range(1, n + 1):
        for tag in tags:
            max_arg = "NN"
            max_value = 0
            for previous_tag in tags:
                emission = emissions[sentence[k - 1]][tag]
                transition = transitions[(previous_tag, tag)]

                pr = pie[k - 1][previous_tag][1] * emission * transition
                if pr > max_value:
                    max_arg = previous_tag
                    max_value = pr
            
            pie[k][tag] = (max_arg, max_value)

    predictions = [None for i in range(n)]
    # Pair shape (I miss typescript!): (<current_tag>, (<previous_tag>, <previous_value>))
    last_step_max = max(pie[n].items(), key=lambda pair: pair[1][1] * transitions[(pair[1], STOP_MARKER)] )
    predictions[n - 1] = last_step_max[0]

    for k in range(n - 2, -1, -1):
        predictions[k] = pie[k + 2][predictions[k + 1]][0]
    
    return predictions
    

def print_viterbi_error_rate(dataset, transitions, emissions):
    known_words_count = 0
    known_hits_count = 0
    unknown_words_count = 0
    unknown_hits_count = 0
    for sentence in dataset:
        predictions = viterbi([pair[0] for pair in sentence], transitions, emissions)
        for i in range(len(sentence)):
            if sentence[i][0] in words_map:
                known_words_count += 1
                if sentence[i][1] == predictions[i]:
                    known_hits_count += 1
            else:
                unknown_words_count += 1
                if sentence[i][1] == predictions[i]:
                    unknown_hits_count += 1
            
    known_words_accuracy = known_hits_count / known_words_count
    unknown_words_accuracy = unknown_hits_count / unknown_words_count
    total_accuracy = (known_hits_count + unknown_hits_count) / (known_words_count + unknown_words_count)
    print("Viterbi Known words prediction err:", 1 - known_words_accuracy)
    print("Viterbi Unknown words prediction err:", 1 - unknown_words_accuracy)
    print("Viterbi Total prediction err:", 1 - total_accuracy)


def calculate_emissions(words_map, tags_histogram, smoothing_count=0):
    emissions = defaultdict(lambda: defaultdict(float))
    for word in words_map:
        for part_of_speech in words_map[word]:
            emissions[word][part_of_speech] = (words_map[word][part_of_speech] + smoothing_count) / (tags_histogram[part_of_speech] + smoothing_count * len(tags_histogram))
    return emissions

def calculate_transitions(dataset):
    transitions = defaultdict(float)
    for sentence in dataset:
        previous_tag = START_MARKER
        for word, tag in sentence:
            transitions[(previous_tag, tag)] += 1
            previous_tag = tag
        transitions[(previous_tag, STOP_MARKER)] += 1
    for transition in transitions:
        transitions[transition] /= len(training_data)
    return transitions


print("Baseline Viterbi:")
print_viterbi_error_rate(test_data, calculate_transitions(training_data), calculate_emissions(words_map, tags_histogram, 0))


# Add one smoothing
print("Add-one Viterbi:")
print_viterbi_error_rate(test_data, calculate_transitions(training_data), calculate_emissions(words_map, tags_histogram, 1))

# Pseudo words
transformed_training_data = transform_pseudo(training_data)
transformed_test_data = transform_pseudo(test_data)
transformed_words_map, transformed_tags_histogram = create_words_map_tags_histogram(transformed_training_data)
print("Viterbi w/ pseudo words")
print_viterbi_error_rate(transformed_test_data, calculate_transitions(transformed_training_data), calculate_emissions(transformed_words_map, transformed_tags_histogram))

# Pseudo words and add one
print("Add-one Viterbi w/ pseudo words")
print_viterbi_error_rate(transformed_test_data, calculate_transitions(transformed_training_data), calculate_emissions(transformed_words_map, transformed_tags_histogram, 1))

# Confusion matrix
# got me confused
