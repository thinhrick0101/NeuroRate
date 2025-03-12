import pyarrow.ipc as ipc
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

file_name = "state.json"

with open(file_name, "r") as f:
    data = json.load(f)
    
data_file_len = len(data["_data_files"])

name = data["_data_files"][0]

with open(name["filename"], "rb") as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()
    
texts = table[1].to_pylist()

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordPieceTrainer(
    vocab_size=30522,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train_from_iterator(texts, trainer)
vocab = tokenizer.get_vocab()

with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(f"{token}\n")


# for i in range (0, data_file_len):
#     name = data["_data_files"][i]
#     with open(name["filename"], "rb") as f:
#         reader = ipc.open_stream(f)
#         table = reader.read_all()
#         sum += len(table[0])
        
        # To extract each column, use indexing from the table
        # rating_array = table[0]
        # text_array = table[1]