import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# LSTM Text Generator - Character Level
# ============================================================

DATA_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/shakespeare.txt"
)

SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 64
LSTM_UNITS = 128
EPOCHS = 10


def load_and_preprocess_text():
    """Download, load and preprocess the Shakespeare dataset."""
    print("Downloading/loading dataset...")

    path = tf.keras.utils.get_file(
        "shakespeare.txt",
        origin=DATA_URL
    )

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Original text length: {len(text)}")

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    print(f"Processed text length: {len(text)}")
    return text


def create_vocabulary(text):
    """Create character-to-index and index-to-character mappings."""
    characters = sorted(set(text))

    char_to_index = {
        char: index for index, char in enumerate(characters)
    }

    index_to_char = {
        index: char for char, index in char_to_index.items()
    }

    return characters, char_to_index, index_to_char


def encode_text(text, char_to_index):
    """Convert characters into integer IDs."""
    return np.array(
        [char_to_index[char] for char in text],
        dtype=np.int32
    )


def create_sequences(encoded_data, sequence_length):
    """Create input sequences and next-character targets."""
    inputs = []
    targets = []

    for i in range(len(encoded_data) - sequence_length):
        inputs.append(
            encoded_data[i:i + sequence_length]
        )
        targets.append(
            encoded_data[i + sequence_length]
        )

    return (
        np.array(inputs, dtype=np.int32),
        np.array(targets, dtype=np.int32)
    )


def build_model(vocab_size):
    """Build the Embedding + LSTM + Dense model."""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            input_length=SEQ_LENGTH
        ),
        LSTM(LSTM_UNITS),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def generate_text(
    model,
    seed_text,
    char_to_index,
    index_to_char,
    num_characters=300
):
    """Generate text iteratively from a seed string."""
    seed_text = seed_text.lower()

    # Keep only characters present in the vocabulary
    seed_text = "".join(
        char for char in seed_text
        if char in char_to_index
    )

    if not seed_text:
        raise ValueError("Seed text contains no known characters.")

    # Pad short seed text to the required sequence length
    if len(seed_text) < SEQ_LENGTH:
        seed_text = (
            " " * (SEQ_LENGTH - len(seed_text))
        ) + seed_text

    generated = seed_text

    for _ in range(num_characters):
        sequence = generated[-SEQ_LENGTH:]

        encoded_sequence = np.array(
            [[char_to_index[char] for char in sequence]],
            dtype=np.int32
        )

        predictions = model.predict(
            encoded_sequence,
            verbose=0
        )

        next_index = int(np.argmax(predictions[0]))
        next_character = index_to_char[next_index]

        generated += next_character

    return generated


def main():
    # 1. Load and preprocess dataset
    text = load_and_preprocess_text()

    # 2. Vocabulary
    characters, char_to_index, index_to_char = create_vocabulary(text)
    vocab_size = len(characters)

    print(f"Vocabulary size: {vocab_size}")

    # 3. Encode text
    encoded_text = encode_text(text, char_to_index)
    print(f"Encoded text shape: {encoded_text.shape}")

    # 4. Create input-output sequences
    X, y = create_sequences(
        encoded_text,
        SEQ_LENGTH
    )

    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # 5. Train/validation split
    split_index = int(len(X) * 0.9)

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # 6. Build model
    model = build_model(vocab_size)
    model.summary()

    # 7. Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # 8. Train
    print("\nStarting model training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    # 9. Save model
    model.save("lstm_text_generator.keras")
    print("\nModel saved as lstm_text_generator.keras")

    # 10. Generate samples
    seed_inputs = [
        "to be or not to be",
        "the king",
        "love is"
    ]

    generated_outputs = []

    for seed in seed_inputs:
        print("\n" + "=" * 70)
        print(f"Seed: {seed}")
        print("=" * 70)

        generated = generate_text(
            model,
            seed,
            char_to_index,
            index_to_char,
            num_characters=300
        )

        print(generated)

        generated_outputs.append(
            f"Seed: {seed}\n\n{generated}\n"
        )

    # 11. Save generated output
    with open(
        "generated_text.txt",
        "w",
        encoding="utf-8"
    ) as file:
        for output in generated_outputs:
            file.write(output)
            file.write("\n" + "=" * 70 + "\n")

    print("\nGenerated text saved to generated_text.txt")
    print("Training and generation completed successfully!")


if __name__ == "__main__":
    main()
