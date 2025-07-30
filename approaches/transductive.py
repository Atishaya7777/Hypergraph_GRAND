import torch
import mlflow
import mlflow.pytorch

from data import ContactDataset, DataSplitter 
from data.dataset import create_hypergraph_dataset
from models import HypergraphGRAND, create_hypergrand_model
from training.trainer import create_hypergraph_trainer

# NOTE: For now, I'm just going to differentiate between my datasets using strings. Later, adapt this to be a factory builder method.
def transductive_learning_approach(dataset_name: str, strategy: str = 'clustering'):
    """
    Transductive learning on each dataset individually
    Args:
        dataset_name: ['contact', 'planetoid', 'planetoid_cora', 'planetoid_citeseer', 'planetoid_pubmed']
        strategy: ['classification', 'clustering']
    """
    print("="*60)
    print("TRANSDUCTIVE LEARNING")
    print("="*60)

    datasets = {}

    dataset_name = dataset_name.lower()

    if(dataset_name == 'contact'):
        datasets.update({
            'contact-high-school': 'datasets/contact-high-school',
            'contact-primary-school': 'datasets/contact-primary-school',
        })
    elif dataset_name == 'planetoid_cora':
        datasets.update({
            'planetoid_cora': 'datasets/cora'
        })
    elif dataset_name == 'planetoid_citeseer':
        datasets.update({
            'planetoid_citeseer': 'datasets/citeseer'
        })
    elif dataset_name == 'planetoid_pubmed':
        datasets.update({
            'planetoid_pubmed': 'datasets/pubmed'
        })


    results = {}

    print(f"Datasets: ", datasets.items())

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

        with mlflow.start_run(run_name=f"Hypergraph GRAND Transductive {dataset_name}"):

            datasetFactory = create_hypergraph_dataset(dataset_name)
            data = datasetFactory.load_data(dataset_path)

            if(dataset_name == 'contact'):
                train_mask = data.train_mask
                val_mask = data.val_mask
                test_mask = data.test_mask
            else:
                train_mask, val_mask, test_mask = DataSplitter.create_transductive_split(
                    data.labels
                )

            if dataset_name.startswith('planetoid'):
                # For Planetoid datasets, use feature dimension
                input_dim = data.node_features.shape[1]
            else:
                # For contact datasets (identity features), use number of nodes
                input_dim = data.num_nodes

            hyperparams = {
                "input_dim": input_dim,
                "hidden_dim": 32,
                "num_layers": 2,
                "alpha": 0.02,
                "dropout": 0.5
            }

            mlflow.log_params(hyperparams)

            scheme_defaults = {
                    "implicit": {"max_iter": 10, "tol": 1e-6},
                    "adaptive": {"min_alpha": 0.01, "max_alpha": 0.5, "tol": 1e-4}
                }

            model = create_hypergrand_model(
                input_dim=hyperparams["input_dim"],
                hidden_dim=hyperparams["hidden_dim"],
                num_layers=hyperparams["num_layers"],
                alpha=hyperparams["alpha"],
                dropout=hyperparams["dropout"],
                scheme="implicit",
                max_iter=15,
                tol=1e-5
            )
            
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            trainer = create_hypergraph_trainer(
                task_type=strategy,
                model=model,
                device=device,
                num_classes=data.num_classes # This will not matter if the strategy is clustering, it'll just not get forwarded to the clustering trainer
            ) 

            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=5e-4)

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


