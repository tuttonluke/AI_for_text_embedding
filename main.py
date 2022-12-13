from transformers import BertModel, BertTokenizer
from word_types.prepositions import prepositions
from time import time
from torch.utils.tensorboard import SummaryWriter
#%%
class BertVisualiser:
    def __init__(self, n_embeddings) -> None:
        self.model_name = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.n_embeddings = n_embeddings
    
    def get_embedding_matrix(self):
        embedding_matrix = self.bert_model.embeddings.word_embeddings.weight.detach()[:self.n_embeddings]
        print(f"Embedding shape: {embedding_matrix.shape}")

        return embedding_matrix
    
    def create_embedding_labels(self):
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
    
    def visualise_embeddings(self):
        writer = SummaryWriter()
        start = time()
        
        embedding_matrix = self.get_embedding_matrix()
        labels, label_names = self.create_embedding_labels()

        writer.add_embedding(
        mat=embedding_matrix,
        metadata=labels,
        metadata_header=label_names
        )
        print(f"Total time: {time() - start}")
        print("Embedding done.")

#%%
if __name__ == "__main__":
    bert = BertVisualiser(n_embeddings=30000)
    bert.visualise_embeddings()



    




