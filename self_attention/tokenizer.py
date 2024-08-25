from typing import List, Dict


class BaseTokenizer:
	def __init__(self, vocab: Dict[str, int]):
		self.vocab = vocab


	@staticmethod
	def tokenize(text: str) -> List[str]:
		tokens = text.replace(",", "").split()
		return tokens

	def encode(self, text: str) -> List[int]:
		tokens = self.tokenize(text)
		ids = [self.vocab[token] for token in tokens]

		return ids