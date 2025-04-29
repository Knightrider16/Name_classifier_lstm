import glob
import os
import string
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def find_files(path): return glob.glob(path)

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_data(data_path='data/*.txt'):
    category_lines = {}
    all_categories = []

    for filename in find_files(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][all_letters.find(letter)] = 1
    return tensor

def name_to_tensor(name):
    return [letter_to_tensor(letter) for letter in name]
