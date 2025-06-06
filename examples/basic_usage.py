"""
Basic usage example for the siamese-confidence package.

This example demonstrates:
1. Training a Siamese model
2. Loading a trained model
3. Computing confidence scores
4. Integration with vector databases
"""

import numpy as np

import siamese_confidence as sc


def main():
    """Demonstrate basic usage of the siamese-confidence package."""

    print("=== Siamese Confidence Scoring Example ===\n")

    # Example 1: Training a model (you would do this once with real data)
    print("1. Training a Siamese model...")
    print("   (This example uses mock data - replace with your embeddings.json)")

    # In practice, you would use:
    # model = sc.train_siamese_model(
    #     train_path="your_embeddings.json",
    #     model_save_path="siamese_model.json",
    #     hidden_layer_sizes=(128, 64),
    #     learning_rate=0.001,
    #     alpha=0.001,
    #     dropout_rate=0.2,
    #     max_iter=300,
    #     verbose=True
    # )

    # For this demo, we'll create a simple model
    model = sc.SiameseModel(
        hidden_layer_sizes=(64, 32), learning_rate=0.01, max_iter=50
    )

    # Create some mock training data (vector differences)
    np.random.seed(42)
    X_train = np.random.randn(100, 384)  # 100 samples, 384D features
    y_train = np.random.randint(0, 2, 100)  # Binary labels

    print("   Training on mock data...")
    model.fit(X_train, y_train, verbose=False)
    print("   ✓ Model trained successfully!")

    # Example 2: Save and load model
    print("\n2. Saving and loading model...")
    model.save_model("demo_model.json")
    print("   ✓ Model saved to demo_model.json")

    loaded_model = sc.load_trained_model("demo_model.json")
    print("   ✓ Model loaded successfully!")

    # Example 3: Compute confidence scores
    print("\n3. Computing confidence scores...")

    # Mock embeddings (in practice, these come from your embedding model)
    page_embedding = np.random.randn(384)
    reference_embeddings = [np.random.randn(384) for _ in range(5)]

    confidence = sc.compute_confidence(
        page_embedding, reference_embeddings, loaded_model
    )

    print(f"   Page embedding confidence: {confidence:.2f}%")

    # Example 4: Batch processing
    print("\n4. Batch confidence computation...")

    batch_embeddings = [np.random.randn(384) for _ in range(3)]
    batch_confidences = []

    for i, emb in enumerate(batch_embeddings):
        conf = sc.compute_confidence(emb, reference_embeddings, loaded_model)
        batch_confidences.append(conf)
        print(f"   Document {i + 1} confidence: {conf:.2f}%")

    avg_confidence = np.mean(batch_confidences)
    print(f"   Average confidence: {avg_confidence:.2f}%")

    # Example 5: Confidence thresholding
    print("\n5. Confidence-based decision making...")

    threshold_high = 80.0
    threshold_low = 60.0

    for i, conf in enumerate(batch_confidences):
        if conf >= threshold_high:
            decision = "AUTO_APPROVE"
        elif conf >= threshold_low:
            decision = "HUMAN_REVIEW"
        else:
            decision = "REJECT"

        print(f"   Document {i + 1}: {conf:.1f}% → {decision}")

    print("\n=== Example completed! ===")
    print(f"Package version: {sc.get_version()}")


class MockVectorDB:
    """Mock vector database for demonstration purposes."""

    def __init__(self):
        self.classes = ["invoice", "receipt", "contract", "report"]

    def search(self, embedding: np.ndarray, top_k: int = 5):
        """Mock search that returns random results."""
        # In practice, this would be your actual vector DB search
        predicted_class = np.random.choice(self.classes)

        # Mock reference embeddings from the same class
        reference_embeddings = [np.random.randn(len(embedding)) for _ in range(top_k)]

        return predicted_class, reference_embeddings


def vector_db_integration_example():
    """Example of integrating with a vector database."""

    print("\n=== Vector Database Integration Example ===\n")

    # Load trained model
    try:
        model = sc.load_trained_model("demo_model.json")
    except FileNotFoundError:
        print("Please run the main() example first to create demo_model.json")
        return

    # Initialize mock vector DB
    vector_db = MockVectorDB()

    # Process a document with multiple pages
    document_embeddings = [
        np.random.randn(384)
        for _ in range(3)  # 3-page document
    ]

    print("Processing multi-page document...")
    page_results = []

    for page_idx, page_emb in enumerate(document_embeddings):
        # 1. Get prediction from vector DB
        predicted_class, references = vector_db.search(page_emb, top_k=5)

        # 2. Compute Siamese confidence
        confidence = sc.compute_confidence(page_emb, references, model)

        page_results.append(
            {
                "page": page_idx + 1,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }
        )

        print(f"   Page {page_idx + 1}: {predicted_class} ({confidence:.1f}%)")

    # Document-level aggregation
    class_votes = {}
    for result in page_results:
        cls = result["predicted_class"]
        if cls not in class_votes:
            class_votes[cls] = []
        class_votes[cls].append(result["confidence"])

    # Get the class with highest average confidence
    best_class = max(class_votes.keys(), key=lambda c: np.mean(class_votes[c]))
    best_confidence = np.mean(class_votes[best_class])

    print("\nDocument classification:")
    print(f"   Final class: {best_class}")
    print(f"   Final confidence: {best_confidence:.1f}%")
    print(f"   Pages processed: {len(document_embeddings)}")


if __name__ == "__main__":
    main()
    vector_db_integration_example()
