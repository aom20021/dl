import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torchaudio
import itertools
import pandas as pd
from datasets import Dataset, DatasetDict
from torchaudio import list_audio_backends
import torch
print(torch.cuda.is_available())  # Should return True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from evaluate import load

# Load the accuracy metric
accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predictions
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

# Path to the dataset folder
data_folder = "genres_original"
genres = ["rock", "reggae", "pop", "metal", "jazz", "hiphop", "disco", "country", "classical", "blues"]

print("Available audio backends:", list_audio_backends())

# Function to load audio files and labels
def load_dataset(data_folder, genres):
    audio_data = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(data_folder, genre)
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                audio_data.append(file_path)
                labels.append(genres.index(genre))  # Encode genre as integer

    return audio_data, labels

# Load data
audio_files, labels = load_dataset(data_folder, genres)

# Function to preprocess audio files
def preprocess_audio(file_path, label):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        return {"audio": waveform.squeeze().numpy(), "label": label}
    except Exception as e:
        print(f"Skipping file {file_path}: {e}")
        return {"audio": None, "label": label}  # Skip file if it can't be processed


# Create Dataset
dataset_dict = {
    "file_path": audio_files,
    "label": labels
}
dataset = Dataset.from_dict(dataset_dict)

# Preprocess the dataset and filter out empty results
preprocessed_dataset = dataset.map(
    lambda x: preprocess_audio(x["file_path"], x["label"]),
    remove_columns=["file_path"]
)
# Filter out empty dictionaries
preprocessed_dataset = preprocessed_dataset.filter(lambda x: "audio" in x and x["audio"] is not None)


# Split into train and test sets
train_test_split = preprocessed_dataset.train_test_split(test_size=0.2)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")


MAX_LENGTH = 16000 * 3  # Max audio length of 3 seconds

# Prepare the dataset for training
def prepare_dataset(batch):
    inputs = feature_extractor(
        batch["audio"],
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_LENGTH,  # Define the maximum length for padding
        truncation=True  # Truncate inputs longer than max_length
    )
    batch["input_values"] = inputs["input_values"]
    batch["labels"] = batch["label"]
    return batch



encoded_dataset = dataset.map(prepare_dataset, remove_columns=["audio"], batched=True, batch_size=16)

print(encoded_dataset)

# Path to save results
results_file = "experiment_results_2_factor.csv"

# Define factors and levels.
batch_sizes = [8]
learning_rates = [1e-4]

# Experimental design: Full factorial
experiments = list(itertools.product(batch_sizes, learning_rates))
results = []

for batch_size, learning_rate in experiments:
    for repetition in range(1):
        print(f"Experiment: batch_size={batch_size}, learning_rate={learning_rate}, repetition={repetition + 1}")

        # Initialize model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            num_labels=len(genres),
            ignore_mismatched_sizes=True
        )

        # Move model to the selected device
        model.to(device)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/bs{batch_size}_lr{learning_rate}_rep{repetition}",
            gradient_checkpointing=True,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,  # Set the learning rate
            per_device_train_batch_size=batch_size,
            num_train_epochs=3,
            save_steps=500,
            save_total_limit=1,
            logging_dir=f"./logs/bs{batch_size}_lr{learning_rate}_rep{repetition}",
            logging_steps=50
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()

        # Log results
        result = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "repetition": repetition + 1,
            "accuracy": metrics["eval_accuracy"],
            "loss": metrics["eval_loss"]
        }
        results.append(result)

        # Save to CSV after each iteration
        df_result = pd.DataFrame([result])
        model.save_pretrained("./saved_model")
        feature_extractor.save_pretrained("./saved_model")
        if not os.path.exists(results_file):
            df_result.to_csv(results_file, index=False)  # Create a new file if it doesn't exist
        else:
            df_result.to_csv(results_file, mode='a', index=False, header=False)  # Append to existing file

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("final_experiment_results_2_factor.csv", index=False)
# Analyze results
print(df_results)


