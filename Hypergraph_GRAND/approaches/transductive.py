import torch

from data import ContactDataset, create_transductive_split
from models import HypergraphGRAND
from training import HypergraphTrainer


def transductive_learning_approach():
    """
    Approach 1: Transductive learning on each dataset individually
    """
    print("="*60)
    print("APPROACH 1: TRANSDUCTIVE LEARNING")
    print("="*60)

    # Dataset paths - adjust these to your actual paths
    datasets = {
        'contact-high-school': 'datasets/contact-high-school',
        'contact-primary-school': 'datasets/contact-primary-school'
    }

    results = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

        # Load dataset
        data = ContactDataset(dataset_path, dataset_name)

        # Create transductive split
        train_mask, val_mask, test_mask = create_transductive_split(
            data.labels)

        # Initialize model
        model = HypergraphGRAND(
            input_dim=data.num_nodes,  # Using node features as input
            hidden_dim=16,
            num_layers=3,
            alpha=0.1,
            dropout=0.1
        )

        # Initialize trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = HypergraphTrainer(model, device)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=1e-5)

        # Train model
        train_results = trainer.train(
            data, train_mask, val_mask, optimizer, num_epochs=10)

        # Evaluate on test set
        test_results = trainer.evaluate(data, test_mask)

        # Store results
        results[dataset_name] = {
            'train_results': train_results,
            'test_results': test_results,
            'dataset_stats': {
                'num_nodes': data.num_nodes,
                'num_hyperedges': data.num_hyperedges,
                'num_classes': data.num_classes
            }
        }

        print(f"\nFinal Results for {dataset_name}:")
        print(
            f"  - Best Val Accuracy: {train_results['best_val_accuracy']:.4f}")
        print(f"  - Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"  - Test Loss: {test_results['test_loss']:.4f}")

    return results
