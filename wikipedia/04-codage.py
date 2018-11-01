# 2018 (c) Nicolas MAUGE, https://github.com/NicolasMauge
import numpy as np
from collections import Counter, defaultdict
import collections
from multiprocessing import Pool
import multiprocessing
from pathlib import Path

import time
from utils.utils import get_num_lines, hms_string, save_vocabulary, load_vocabulary
from tqdm import tqdm

max_vocab = 80000 # 60000 seems not enough for french (conjugaisons, accords, etc.)


def load_data(file, lower=True):
	"""
	load data 1000 lines at a time
	"""
	with open(file, "r") as file:
		data = []
		for index, line in enumerate(file):
			if lower:
				line = line.rstrip().lower().split(";")
			else:
				line = line.rstrip().split(";")
			data.append(line)
			
			if len(data) == 1000:
				yield data
				data=[]
			
		yield data

def processing_data_counter(data):
	"""
	for a worker, count the words
	"""
	data_counter = Counter()
	for line in data:
		data_counter += Counter(line)

	return data_counter

def processing_data_transcode(data):
	"""
	for a worker, transcode the sentences with the vocabulary 'stoi' (sentence to index)
	"""
	data_coded = []
	stoi_copy = stoi.copy()
	up = str(stoi["_up_"])
	for line in data:
		line_coded = []
		for token in line:
			token_coded = stoi_copy[token.lower()]
			if token_coded!=0 and token.isupper():
				line_coded.append(up)
			line_coded.append(str(token_coded))

		data_coded.append(";".join(line_coded))	

	return data_coded


def count_words(vocab_stoi, vocab_itos):
	c = Counter()
	file = "data/articles/cleaned_text_tok.csv"
	print(f"Word freq count in {file}:")
	
	start_time = time.time()
	iterator = load_data(file)
	with multiprocessing.Pool() as pool:
		for result in tqdm(pool.imap_unordered(processing_data_counter, iterator), total=get_num_lines(file)//1000, ascii=True):
			c += result

	elapsed_time = time.time() - start_time
	print("- Elapsed time: {}".format(hms_string(elapsed_time)))

	most_common = ["_unk_", "_up_"]+[word for word,c in c.most_common(max_vocab) if c>2]
	with open("test.csv", "w") as filename:
		filename.write(";".join(most_common))

	stoi = {v:k for k,v in enumerate(most_common)}
	itos = {k:v for k,v in enumerate(most_common)}
	print(itos)

	print("- Saving vocabularies")
	save_vocabulary(vocab_stoi, stoi)
	save_vocabulary(vocab_itos, itos)
	return stoi, itos



def transcode_text():
	file = "data/articles/cleaned_text_tok.csv"
	print(f"Trancoding {file}:")
	
	start_time = time.time()
	iterator = load_data(file, lower=False)
	with multiprocessing.Pool() as pool:
		with open(file[:-4]+"_coded.csv", "w") as filename:
			for result in tqdm(pool.map(processing_data_transcode, iterator), total=get_num_lines(file)//1000, ascii=True):
				for sentence in result:
					filename.write(sentence)
				filename.write("\n")

	elapsed_time = time.time() - start_time
	print(f"- Elapsed time: {hms_string(elapsed_time)}")

def exclude_articles():
	"""
	Exclude articles with too much "0" (=too many words with little freq)
	"""
	file = "data/articles/cleaned_text_tok_coded.csv"

	n_articles_excluded = 0
	with open(file[:-4]+"_ex.csv", "w") as dest:
		with open(file, "r") as data:
			for article in data:
				words = article.split(";")
				n_words = len(words)
				n_0 = words.count("0")
				if n_0 / n_words < 0.1:
					dest.write(article + "\n")
				else:
					n_articles_excluded +=1


	print(f"Article excluded: {n_articles_excluded}")


def vocabulary_transcode():
	vocab_stoi = "data/articles/vocab_stoi.pkl"
	vocab_itos = "data/articles/vocab_itos.pkl"
	global stoi, itos
	if not Path(vocab_stoi).is_file() or not Path(vocab_itos).is_file():
		stoi, itos = count_words(vocab_stoi, vocab_itos)
	else:
		stoi, itos = load_vocabulary(vocab_stoi), load_vocabulary(vocab_itos)

	stoi = defaultdict(int, stoi)

	transcode_text()
	
	exclude_articles()

if __name__ == '__main__':
	vocabulary_transcode()
	
