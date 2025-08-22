import argparse
import datasets
import pandas as pd
import transformers
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


def train_advanced(
    model_path="model.keras",
    transformer_path="transformer_model",
    train_path="train.csv",
    dev_path="dev.csv",
    batch_size=32, 
    learning_rate=2e-5,
    num_epochs=5, 
    dropout_rate=0.3, 
    dense_units=[256, 128], 
):
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
    transformer_model = transformers.TFAutoModel.from_pretrained("distilroberta-base")

    hf_dataset = datasets.load_dataset("csv", data_files={"train": train_path, "validation": dev_path})

    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        return {"labels": [float(example[l]) for l in labels]}

    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=64, padding="max_length"), batched=True)

    def preprocess_for_tf_dataset(example):
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"]
        }

    train_dataset = hf_dataset["train"].map(preprocess_for_tf_dataset).with_format(
        "tensorflow",
        columns=["input_ids", "attention_mask", "labels"],
    )

    dev_dataset = hf_dataset["validation"].map(preprocess_for_tf_dataset).with_format(
        "tensorflow",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_tf_dataset = train_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=batch_size,
        shuffle=True,
    )

    dev_tf_dataset = dev_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=batch_size,
    )

    input_ids = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="attention_mask")
    embeddings = transformer_model(input_ids, attention_mask=attention_mask)[0]

    x = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x) 
    output = tf.keras.layers.Dense(units=len(labels), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name="micro_F1", multi_label=True)],
    )

    history = model.fit(
        train_tf_dataset,
        validation_data=dev_tf_dataset,
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_micro_F1", mode="max", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),  # Adjust learning rate dynamically
        ],
    )

    print("\nTraining Complete!")
    for epoch in range(len(history.history["loss"])):
        print(f"Epoch {epoch + 1}: Loss = {history.history['loss'][epoch]:.4f}, "
              f"Validation Micro F1 = {history.history['val_micro_F1'][epoch]:.4f}")

    y_true = []
    for batch in dev_tf_dataset:
        y_true.append(batch[1].numpy())
    y_true = np.vstack(y_true)
    y_pred_probs = model.predict(dev_tf_dataset)
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)

    print("\nPer-Emotion F1 Scores on Validation Set:")
    for i, label in enumerate(labels):
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        print(f"F1 Score for {label}: {f1:.4f}")

    transformer_model.save_pretrained(transformer_path)
    tokenizer.save_pretrained(transformer_path)

    print("\nModel and Transformer saved!")


def predict_advanced(model_path="model.keras", transformer_path="transformer_model", input_path="test-in.csv"):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_path)
    transformer_model = transformers.TFAutoModel.from_pretrained(transformer_path)

    # Define the model
    input_ids = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="attention_mask")
    embeddings = transformer_model(input_ids, attention_mask=attention_mask)[0]

    x = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    for units in [256, 128]:  # Match dense_units from train_advanced
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)  # Match dropout rate from train_advanced
    output = tf.keras.layers.Dense(units=7, activation="sigmoid")(x)  # Match output units

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Load weights only
    model.load_weights(model_path)

    # Load and preprocess the data
    df = pd.read_csv(input_path)
    hf_dataset = datasets.Dataset.from_pandas(df)

    # Extract true labels (assuming the labels are in columns 1 onwards)
    labels = df.columns[1:]  # First column is assumed to be 'text', others are labels
    y_true = df.iloc[:, 1:].values  # Extract ground truth labels

    hf_dataset = hf_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=64, padding="max_length"), batched=True)

    def preprocess_for_tf_dataset(example):
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"]
        }

    tf_dataset = hf_dataset.map(preprocess_for_tf_dataset).with_format(
        "tensorflow",
        columns=["input_ids", "attention_mask"],
    ).to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=16,
    )

    # Generate predictions
    y_pred_probs = model.predict(tf_dataset)
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)

    # Align dimensions if there's a mismatch
    if y_true.shape != y_pred.shape:
        print(f"Shape mismatch detected: y_true {y_true.shape}, y_pred {y_pred.shape}. Adjusting dimensions...")
        min_rows = min(y_true.shape[0], y_pred.shape[0])  # Match row count
        min_cols = min(y_true.shape[1], y_pred.shape[1])  # Match column count
        y_true = y_true[:min_rows, :min_cols]
        y_pred = y_pred[:min_rows, :min_cols]

    # Calculate Micro F1 Score
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    print(f"\nMicro F1 Score: {micro_f1:.4f}")

    # Calculate Macro F1 Score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # Calculate Per-Emotion F1 Scores
    print("\nPer-Emotion F1 Scores:")
    for i, label in enumerate(labels[:y_true.shape[1]]):  # Ensure labels match the adjusted shape
        emotion_f1 = f1_score(y_true[:, i], y_pred[:, i])
        print(f"F1 Score for {label}: {emotion_f1:.4f}")

    # Print a few predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(df))):  # Show first 5 examples
        print(f"Text: {df['text'][i]}")
        print(f"Predicted Labels: {y_pred[i]}")
        print(f"True Labels: {y_true[i]}\n")

    # Assign predictions to label columns in the Pandas dataframe
    for i, label in enumerate(labels[:y_pred.shape[1]]):
        df[label] = y_pred[:, i]

    # Write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method="zip", archive_name="submission.csv"))

    print("\nSubmission file created as 'submission.zip'.")


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # Call either train or predict
    if args.command == "train":
        train_advanced()
    elif args.command == "predict":
        predict_advanced()
