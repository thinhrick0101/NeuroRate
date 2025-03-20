# NeuroRate

NeuroRate is a machine learning project designed to predict star ratings for Amazon reviews using a BERT-based model.

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run:
```bash
python main.py --mode train
```

### Inference

To predict the star rating for a given text, run:
```bash
python main.py --mode inference --text "This product is amazing!"
```

### Exploring the Dataset

To explore the dataset structure, run:
```bash
python main.py --mode explore_dataset
```

## Project Structure

- `batch_inference.py`: Script for running batch predictions on a list of texts.
- `config.py`: Configuration settings for the model, tokenizer, and training.
- `dataset.py`: Dataset handling and DataLoader creation.
- `inference.py`: Inference class for predicting star ratings.
- `main.py`: Main script for training, inference, and dataset exploration.
- `model.py`: BERT model definition.
- `process.py`: Dataset processing functions.
- `tokenizer.py`: Tokenizer training and loading.
- `training.py`: Training loop and evaluation functions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
