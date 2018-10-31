# 2018 (c) Nicolas MAUGE, https://github.com/NicolasMauge
import os
import time
from clean_wiki.wiki_fr_basic_clean import clean_text
from clean_wiki.wiki_xml_to_text import process_xml_dump
from utils.utils import hms_string
from tqdm import tqdm

out_dir = "data/articles/"
ENCODING = "utf-8"


def get_corpus_text(n_tokens=150000000):
	file = "cleaned_text.txt"
	iter_dump = process_xml_dump()
	
	n_words = 0
	n_articles_collected = 0
	print(f"Collecting words:")
	pbar = tqdm(total=n_tokens, ascii=True, unit="words")
	with open(out_dir+file, "w") as articles:
		while n_words < n_tokens: # fastai : 100 000 000 tokens
			# no need to clean everything for a corpus dedicated to machine learning
			try:
				text = next(iter_dump)
			except StopIteration:
				print("The end of the XML dump has been reached")
				break

			n_lines, cleaned_text = clean_text(text)

			if n_lines > 50: # since this text will be used for ML
				n_articles_collected = n_articles_collected+1
				delta_words = len(cleaned_text.split(" "))
				n_words = n_words + delta_words
				articles.write(cleaned_text + "\n")
				pbar.update(delta_words)	

		print(f"{n_words} tokens collected in {n_articles_collected} articles")
		pbar.close()


if __name__ == '__main__':
	start_time = time.time()
	get_corpus_text()
	elapsed_time = time.time() - start_time
	print("Elapsed time: {}".format(hms_string(elapsed_time)))