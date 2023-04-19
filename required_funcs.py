"""we use this file to store all the functions that we will use in the main file"""


# we use this function to find the majority element in a list
def find_majority(k):
    """Return the element that occurs most frequently in the list k."""
    myMap = {}
    maximum = ("", 0)  # (occurring element, occurrences)
    for n in k:
        if n in myMap:
            myMap[n] += 1
        else:
            myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]:
            maximum = (n, myMap[n])

    return maximum


# we need to define a function that will take the labels and give the max occuring label
# TODO: tie breaking
def majority_vote(labels):
    """Return the majority vote of the labels"""
    majority = []
    for label in labels:
        majority.append(find_majority(label)[0])
    return majority


def label_convertor(labels):
    """Converts the labels to 0 and 1"""
    return [0 if label == "NO" else 1 for label in labels]


def label_deconvertor(labels):
    """Converts the labels to NO and YES"""
    return ["NO" if label == 0 else "YES" for label in labels]


def split_lang(texts, langs, labels):
    """Splits the texts into english and spanish"""
    espanol, esp_labels = [], []
    english, eng_labels = [], []
    for text, lang, label in zip(texts, langs, labels):
        if lang == "es":
            espanol.append(text)
            esp_labels.append(label)
        else:
            english.append(text)
            eng_labels.append(label)
    return english, eng_labels, espanol, esp_labels
