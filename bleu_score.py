import numpy as np
import nltk
from nltk.util import ngrams
from collections import Counter
nltk.download('punkt')


class Bleu():
  """
  
  """
  def __init__(self, n, n_gram_weights):
    self.n = n
    self.n_gram_weights = n_gram_weights


  def brevity_penalty(self, candidate, reference):
    """
    Compute brevity score given candidate and reference text

    """
    if len(candidate) > len(reference):
      return 1
    else:
      return np.exp(1 - len(reference)/ len(candidate))

      


  def precision(self, candidate, reference):
    """
    Compute clipped precision score
    """

    w = len(candidate)

    scores = np.array([])

    for i in range(1, self.n+1):

      cand_n_gram = Counter(ngrams(candidate, i))
      ref_n_gram = Counter(ngrams(reference, i))

      w = sum(cand_n_gram.values())
      

      for j in cand_n_gram:
        if j in ref_n_gram:
          if cand_n_gram[j] > ref_n_gram[j]:
            cand_n_gram[j] = ref_n_gram[j]
        else:
          cand_n_gram[j] = 0
        score = np.sum(list(cand_n_gram.values()))/w

      scores = np.append(scores, score)

    if self.n_gram_weights is None:
      self.n_gram_weights = np.array([0.25]*self.n)
    b = np.exp(np.multiply(self.n_gram_weights, np.log(scores)))
    return np.sum(b)


  @staticmethod
  def bleu(precision, brevity):
    "Compute bleu score from brevity and precision scores"
    
    return brevity * precision

  def __call__(self, candidate:str, reference:str):
    
    candidate_tokenized, reference_tokenized = self.preprocess(candidate, reference)

    precision = self.precision(candidate_tokenized, reference_tokenized)

    brevity_penalty = self.brevity_penalty(candidate_tokenized, reference_tokenized)

    return self.bleu(precision, brevity_penalty)*100


  @staticmethod
  def preprocess(candidate:str, reference:str):
    ref = nltk.word_tokenize(reference.lower())
    cand = nltk.word_tokenize(candidate.lower())
    return cand, ref



# bleu = Bleu()


