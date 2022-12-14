#%%
from time import time
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer
from transformers import logging
from word_types.prepositions import prepositions
import torch
import torchmetrics
#%%
class BertVisualiser:
    def __init__(self, n_embeddings) -> None:
        self.model_name = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.n_embeddings = n_embeddings
        self.embedding_matrix = self.get_embedding_matrix()
        self.labels, self.label_names = self.create_embedding_labels()
        
    def get_embedding_matrix(self) -> torch.Tensor:
        """Returns a matrix of pretrained word embeddings.

        Returns
        -------
        torch.Tensor
            Torch tensor with size [n_embeddings, 768].
        """
        embedding_matrix = self.bert_model.embeddings.word_embeddings.weight.detach()[:self.n_embeddings]
        # print(f"Embedding shape: {embedding_matrix.shape}")

        return embedding_matrix
    
    def create_embedding_labels(self) -> tuple:
        """Creates labels and label names, based on the dictionary of functions defined in
        label_functions, by which words may be categorized and sortedin the tensorboard 
        visualisation.

        Returns
        -------
        tuple of lists
            tuple of two lists, one of labels and one of label names.
        """
        label_functions = {
            "Length" : lambda word: len(word),
            "# Vowels" : lambda word: len([char for char in word if char in "aeiou"]),
            "Is Number?" : lambda word: word.isdigit(),
            "Is Preposition?" : lambda word: word in prepositions
        }
        labels = [
                [word, 
                *[label_function(word) for label_function in label_functions.values()]]
                for word in list(self.bert_tokenizer.ids_to_tokens.values())[:self.n_embeddings]
                ]
        label_names = ["Word", *list(label_functions.keys())]

        return labels, label_names

    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Gets the embedding vector of the input word.

        Parameters
        ----------
        word : str
            Word whos embedding is to be returned.

        Returns
        -------
        torch.Tensor
            Embedding of the queried word.
        """
        self.bert_tokenizer.tokens_to_ids = {
            token : id for id, token in self.bert_tokenizer.ids_to_tokens.items() 
        }
        token_id = self.bert_tokenizer.tokens_to_ids[word]
        word_embedding = self.embedding_matrix[token_id]

        return word_embedding
    
    def get_token_from_embedding(self, embedding, n=20):
        similarity = torchmetrics.functional.pairwise_cosine_similarity(
            embedding.unsqueeze(0), self.embedding_matrix).squeeze()
        similarity_idx = reversed(torch.argsort(similarity, dim=0))[:n]

        return [list(self.bert_tokenizer.ids_to_tokens.values())[idx] for idx in similarity_idx]

    def analogy_solver(self, a:str, b:str, c:str, n=2):
        """Compares words with the logic "a is to b as c is to d" where d is found in the
        embedding matrix. E.g. london is to england as madrid is to spain.
        """
        a_embedding = self.get_word_embedding(a)
        b_embedding = self.get_word_embedding(b)

        transformation_vector = b_embedding - a_embedding

        c_embedding = self.get_word_embedding(c)
        d_embedding = c_embedding + transformation_vector
        nearest_tokens = self.get_token_from_embedding(d_embedding, n=n+1)        
        for d in nearest_tokens:
            if d == b:
                continue
            elif c == d:
                continue
            print(f"{a} is to {b} as {c} is to {d}")
        self.embedding_matrix = torch.cat(
            (
                self.embedding_matrix,
                transformation_vector.unsqueeze(0)
            )
        )
        self.labels = [
            *self.labels,
            ["INTERPOLATION_VECTOR", 0, 0, 0, 0]
        ]
    
    def visualise_embeddings(self):
        """Visualise the embeddings using Tensorboard.
        """
        writer = SummaryWriter()
        start = time()

        writer.add_embedding(
        mat=self.embedding_matrix,
        metadata=self.labels,
        metadata_header=self.label_names
        )
        print(f"Total time: {time() - start}")
        print("Embedding done.")
#%%
if __name__ == "__main__":
    logging.set_verbosity_error() # removes annnoying warning
    bert = BertVisualiser(n_embeddings=30000)
    bert.analogy_solver("swim", "swam", "run")
    bert.visualise_embeddings()

# TODO average of list of words
# TODO list of words to add together and see what results (e.g. man + royalty = king?)
# TODO try other pretrained embeddings
# TODO fine tune bert on our own dataset (use the unused) 
