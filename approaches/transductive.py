import torch
import mlflow
import mlflow.pytorch

from data import ContactDataset, create_transductive_split
from models import HypergraphGRAND
from training import HypergraphTrainer


def transductive_learning_approach():
    """
    Transductive learning on each dataset individually
    """
    print("="*60)
    print("TRANSDUCTIVE LEARNING")
    print("="*60)

    datasets = {
        'contact-high-school': 'datasets/contact-high-school',
        'contact-primary-school': 'datasets/contact-primary-school'
    }

    results = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

        with mlflow.start_run(run_name=f"Hypergraph GRAND Transductive {dataset_name}"):

            data = ContactDataset(dataset_path, dataset_name)

            train_mask, val_mask, test_mask = create_transductive_split(
                data.labels)

            hyperparams = {
                "input_dim": data.num_nodes,
                "hidden_dim": 16,
                "num_layers": 3,
                "alpha": 0.1,
                "dropout": 0.1
            }

            mlflow.log_params(hyperparams)

            model = HypergraphGRAND(
                input_dim=hyperparams["input_dim"],
                hidden_dim=hyperparams["hidden_dim"],
                num_layers=hyperparams["num_layers"],
                alpha=hyperparams["alpha"],
                dropout=hyperparams["dropout"]
            )

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            trainer = HypergraphTrainer(model, device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.01, weight_decay=1e-5)

            train_results = trainer.train(
                data, train_mask, val_mask, optimizer, num_epochs=100)

            test_results = trainer.evaluate(data, test_mask)

            mlflow.pytorch.log_model(model, artifact_path="model")

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
