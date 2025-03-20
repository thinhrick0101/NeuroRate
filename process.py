import pyarrow.ipc as ipc
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
<<<<<<< HEAD
from datasets import load_dataset
from config import Config

def process_dataset(ds_raw, config:Config):
    
    SIZE = config.max_examples
    SIZE_1 = SIZE * 0.33
    SIZE_2 = SIZE * 0.15
    SIZE_3 = SIZE * 0.33
    SIZE_4 = SIZE * 0.2
    SIZE_5 = SIZE * 0.33
    
    texts = ds_raw["text"]
    ratings = ds_raw["rating"]

    dict = []
    one_star = 0
    two_star = 0
    three_star = 0
    four_star = 0
    five_star = 0

    for rating, text in zip(ratings, texts):
        if (int(rating) == 1) and (one_star < SIZE_1):
            one_star = one_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        # elif (int(rating) == 2) and (two_star < SIZE_2):
        #     two_star = two_star + 1
        #     item = {"text": text, "rating": rating}
        #     dict.append(item)
        elif (int(rating) == 3) and (three_star < SIZE_3):
            three_star = three_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        # elif (int(rating) == 4) and (four_star < SIZE_4):
        #     four_star = four_star + 1
        #     item = {"text": text, "rating": rating}
        #     dict.append(item)
        elif (int(rating) == 5) and (five_star < SIZE_5):
            five_star = five_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        # elif (one_star == SIZE_1 and two_star == SIZE_2 and three_star == SIZE_3 and four_star == SIZE_4 and five_star == SIZE_5):
        #     break
        
        elif (one_star == SIZE_1 and three_star == SIZE_3 and five_star == SIZE_5):
            break
    return dict
=======

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
>>>>>>> 59bb0560b21d2b9490b62232c656dee6ef39edc3
