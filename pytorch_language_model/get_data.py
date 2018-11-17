"""
This program will take three files (train / valid / test) and 
return three iterators :
						n_batch-->
		articles
			|			
			|
			V
"""
import random
import numpy as np
from utils.utils import get_num_lines, hms_string, save_vocabulary, load_vocabulary
import time

import torch

bptt = 70

def load_data(filename):
	with open(filename, "r") as file:
		for article in file:
			yield article

def load_data_np(filename):
	for article in load_data(filename):
		yield np.array(article.rstrip().split(";"))

class get_data():
	def __init__(self, bptt, filenames, n_batch=64, phase="train"):
		self.bptt = bptt
		self.phases = ["train", "valid", "test"]
		self.n_batch = n_batch
		self.set_phase(phase)

		if isinstance(filenames, dict):
			if [data_set in filenames for data_set in self.phases] == [True]*3:
				self.filenames = filenames
			else:
				raise ValueError(f"The parameter 'filenames'={filenames} don't have the 'train', 'valid' and 'test' keys")
		else:
			raise ValueError(f"The parameter 'filenames'={filenames} is not a dictionnary")
		
		data=[]
		for article in load_data_np(self.filenames[phase]):
			data.append(article)
		self.data = data

		self.init_mini_batch()
		self.iterator = self._iterator()

	def set_phase(self, phase):
		if phase in self.phases:
			self.phase=phase
		else:
			raise ValueError(f"Phase parameter (={phase}) must be 'train', 'test' or 'valid'")

	@staticmethod
	def split(filename, proportions=[0.7, 0.15, 0.15]):
		"""
		This function splits the main file in 3 files (train / valid / test)
		"""
		with open(filename[:-4]+"_train.csv", "w") as file_train:
			with open(filename[:-4]+"_valid.csv", "w") as file_valid:
				with open(filename[:-4]+"_test.csv", "w") as file_test:
					for article in load_data(filename):
						p = np.random.uniform()
						if p<proportions[0]:
							file_train.write(article)
						elif p<proportions[0]+proportions[1]:
							file_valid.write(article)
						else:
							file_test.write(article)

	def len_seq(self):
		# https://github.com/salesforce/awd-lstm-lm/blob/master/main.py
		rand_bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
		return max(5, int(np.random.normal(rand_bptt, 5)))

	def shuffle(self):
		permutation = np.random.permutation(len(self.data))
		self.data = [self.data[i] for i in permutation]

	def queue_articles(self):
			for article in self.data:
				yield article
	
	def init_mini_batch(self):
		self.shuffle()
		self.queue = self.queue_articles()


	def get_mini_batch(self):
		while True:
			try:
				batchs = [next(self.queue) for _ in range(self.n_batch)]
			except StopIteration:
				break

			min_len = min([article.shape[0] for article in batchs])
			yield torch.from_numpy(np.stack([article[:min_len].astype(int) for article in batchs]).T).contiguous()
	

	def __iter__(self):
		return self

	def __next__(self):
		return next(self.iterator)

	def _iterator(self):
		index = 0
		for b in self.get_mini_batch():
			while True:
				seq_len = self.len_seq()
				if index+seq_len+1 <= b.shape[0]:
					yield b[index:index+seq_len], b[index+1:index+seq_len+1].view(-1)
				else:
					break
				index+=seq_len


if __name__ == '__main__':
	filenames = {"train":"data/wiki_text_train.csv", "test":"data/wiki_text_test.csv", "valid":"data/wiki_text_valid.csv"}
	data_class = get_data(70, filenames, n_batch=64, phase="valid")

