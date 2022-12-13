from time import time
from torch.utils.tensorboard import SummaryWriter

def visualise_embeddings(embeddings, labels=None, label_names="label"):
    """_summary_

    Parameters
    ----------
    embeddings : _type_
        _description_
    labels : _type_, optional
        _description_, by default None
    label_names : str, optional
        _description_, by default "label"
    """
    print("Embedding")

    writer = SummaryWriter()
    start = time()
    writer.add_embedding(
        mat=embeddings,
        metadata=labels,
        metadata_header=label_names
    )
    print(f"Total time: {time() - start}")
    print("Embedding done.")