"""
Example usage of the Siamese MLP model for confidence scoring in an application.
This shows how to integrate the trained model with your existing vector database workflow.
"""

import numpy as np
from siamese_trainer import compute_confidence, load_trained_model


class DocumentClassifier:
    """Example application class that uses Siamese confidence scoring."""

    def __init__(self, siamese_model_path: str):
        """Initialize with a trained Siamese model."""
        self.siamese_model = load_trained_model(siamese_model_path)

    def classify_document_with_confidence(
        self, page_embeddings: list, vector_db_retrieval_func, k_neighbors: int = 5
    ) -> dict:
        """
        Classify a document and compute confidence using the Siamese method.

        Args:
            page_embeddings: List of numpy arrays, one per page
            vector_db_retrieval_func: Function that takes an embedding and returns
                                    (class_name, list_of_reference_embeddings)
            k_neighbors: Number of neighbors to retrieve from vector DB

        Returns:
            Dict with classification results and confidence scores
        """
        page_results = []
        class_votes = {}

        # Process each page
        for i, page_emb in enumerate(page_embeddings):
            # Get the predicted class and reference embeddings from vector DB
            pred_class, ref_embeddings = vector_db_retrieval_func(page_emb, k_neighbors)

            # Compute Siamese confidence for this page
            page_confidence = compute_confidence(
                page_emb, ref_embeddings, self.siamese_model
            )

            page_results.append(
                {
                    "page_index": i,
                    "predicted_class": pred_class,
                    "confidence": page_confidence,
                }
            )

            # Collect votes for document-level classification
            if pred_class not in class_votes:
                class_votes[pred_class] = []
            class_votes[pred_class].append(page_confidence)

        # Document-level aggregation
        doc_confidences = {}
        for class_name, confidences in class_votes.items():
            # Average confidence for this class across pages
            doc_confidences[class_name] = np.mean(confidences)

        # Pick the class with highest average confidence
        if doc_confidences:
            best_class = max(doc_confidences.keys(), key=lambda c: doc_confidences[c])
            best_confidence = doc_confidences[best_class]
        else:
            best_class = None
            best_confidence = 0.0

        return {
            "document_class": best_class,
            "document_confidence": best_confidence,
            "class_confidences": doc_confidences,
            "page_results": page_results,
            "num_pages": len(page_embeddings),
        }


def mock_vector_db_retrieval(embedding: np.ndarray, k: int = 5):
    """
    Mock function simulating vector database retrieval.
    In your real application, this would query your actual vector DB.

    Args:
        embedding: Query embedding
        k: Number of neighbors to return

    Returns:
        Tuple of (predicted_class, list_of_reference_embeddings)
    """
    # Mock prediction - in reality this would come from your vector DB search
    classes = ["invoice", "receipt", "contract", "report"]
    predicted_class = np.random.choice(classes)

    # Mock reference embeddings - in reality these would come from your vector DB
    # These should be embeddings from the same class as the prediction
    reference_embeddings = [np.random.random(len(embedding)) for _ in range(k)]

    return predicted_class, reference_embeddings


def real_vector_db_retrieval_template(embedding: np.ndarray, k: int = 5):
    """
    Template for real vector database retrieval function.
    Replace this with your actual vector DB implementation.
    """
    # Example using a hypothetical vector DB client
    # results = vector_db_client.search(
    #     vector=embedding.tolist(),
    #     top_k=k,
    #     include_metadata=True
    # )
    #
    # # Get the most common class from top results
    # classes = [result['metadata']['class'] for result in results]
    # predicted_class = max(set(classes), key=classes.count)
    #
    # # Get embeddings from the same class
    # same_class_results = [r for r in results if r['metadata']['class'] == predicted_class]
    # reference_embeddings = [np.array(r['vector']) for r in same_class_results]
    #
    # return predicted_class, reference_embeddings

    # For now, use the mock function
    return mock_vector_db_retrieval(embedding, k)


if __name__ == "__main__":
    # Step 1: Train the Siamese MLP model (do this once)
    print("Training Siamese MLP model...")
    from siamese_trainer import train_siamese_model

    model = train_siamese_model(
        train_path="../data/embeddings/train_text_embeddings.json",
        model_save_path="siamese_mlp_model.json",
        hidden_layer_sizes=(128, 64),
        learning_rate=0.001,
        alpha=0.001,
        dropout_rate=0.2,
        max_iter=300,
        verbose=True,
    )

    # Step 2: Use the trained model in your application
    print("\n" + "=" * 50)
    print("Using trained model for classification...")

    # Initialize classifier
    classifier = DocumentClassifier("siamese_mlp_model.json")

    # Example document with 3 pages (mock embeddings)
    doc_embeddings = [
        np.random.random(384),  # Page 1
        np.random.random(384),  # Page 2
        np.random.random(384),  # Page 3
    ]

    # Classify with confidence
    results = classifier.classify_document_with_confidence(
        page_embeddings=doc_embeddings,
        vector_db_retrieval_func=mock_vector_db_retrieval,
        k_neighbors=5,
    )

    # Print results
    print("\nDocument Classification Results:")
    print(f"  Predicted Class: {results['document_class']}")
    print(f"  Document Confidence: {results['document_confidence']:.2f}%")
    print(f"  Number of Pages: {results['num_pages']}")

    print("\nPer-Class Confidences:")
    for class_name, confidence in results["class_confidences"].items():
        print(f"  {class_name}: {confidence:.2f}%")

    print("\nPer-Page Results:")
    for page_result in results["page_results"]:
        print(
            f"  Page {page_result['page_index']}: "
            f"{page_result['predicted_class']} "
            f"({page_result['confidence']:.2f}%)"
        )

    # Example of using confidence for thresholding
    confidence_threshold = 75.0
    if results["document_confidence"] >= confidence_threshold:
        print(
            f"\n✅ Auto-approved (confidence {results['document_confidence']:.1f}% >= {confidence_threshold}%)"
        )
    else:
        print(
            f"\n⚠️  Manual review needed (confidence {results['document_confidence']:.1f}% < {confidence_threshold}%)"
        )
