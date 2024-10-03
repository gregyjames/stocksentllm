import os
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import pandas as pd
import torch

MODEL_DIR = "./saved_model"  # Directory where the model will be saved
BASE_MODEL = "distilbert-base-uncased"
CSV_DATA = "all-data.csv"  # From kaggle
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"


def main():
    if not os.path.isfile(CSV_DATA):
        print("Training data not found, exiting...")
        return

    # Use the model for inference
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Load CSV data into a pandas DataFrame
    data = pd.read_csv(
        CSV_DATA, delimiter=",", encoding="latin-1", names=["sentiment", "title"]
    )
    print(data.columns)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    categorical_sentiment = pd.Categorical(data["sentiment"])

    # Get the categories (unique values in the sentiment column)
    categories = categorical_sentiment.categories
    print("Categories (Labels):", categories)

    # Get the corresponding integer codes
    codes = categorical_sentiment.codes

    mapping = dict(zip(categories, range(len(categories))))
    print("Category-to-Code Mapping:", mapping)

    labels = pd.Categorical(
        data["sentiment"]
    ).codes  # Convert sentiment to integer codes

    # Define a preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples,  # Apply the tokenizer on the title (your text data)
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    # Apply the tokenizer to your text data
    tokenized_data = preprocess_function(data["title"].tolist())

    # Convert to Hugging Face Dataset format and add the labels
    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
            "labels": labels,
        }
    )

    # Perform train-test split
    train_testsplit = dataset.train_test_split(
        test_size=0.2, seed=42
    )  # 80% train, 20% test
    train_dataset = train_testsplit["train"]
    eval_dataset = train_testsplit["test"]

    # Check if the model directory exists
    if os.path.exists(MODEL_DIR):
        print(f"Loading model from {MODEL_DIR}...")
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        print(f"Training a new model and saving it to {MODEL_DIR}...")

        # Load the DistilBERT model for sequence classification with the appropriate number of labels
        model = DistilBertForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(set(labels))
        )

    # Ensure the model is moved to the correct device
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        use_mps_device=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Pass the training dataset
        eval_dataset=eval_dataset,  # Pass the evaluation dataset
    )

    trainer.train()

    # Evaluate the model (optional)
    print("Evaluating the model...")
    result = trainer.evaluate()  # Uncomment if you want to perform evaluation
    print(result)

    # Example sentence for inference
    example_sentence = "OpenAI Just Gave Spectacular News to Nvidia Stock Investors"

    # Tokenize the example sentence
    inputs = tokenizer(
        example_sentence,
        return_tensors="pt",  # Use pytorch tensors
        truncation=True,
        padding=True,
        max_length=128,
    )

    # Move the input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Inference (ensure no gradients are calculated)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and make predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Print the predicted label
    print(f"Prediction for the sentence: {predictions.item()}")


if __name__ == "__main__":
    main()
