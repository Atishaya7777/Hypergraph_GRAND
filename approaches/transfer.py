import torch
import mlflow
import mlflow.pytorch

from data import ContactDataset, DataSplitter 
from models import HypergraphGRAND
from training import HypergraphTrainer

def transfer_learning_approach():
    """
    Transfer learning from primary school to high school
    """
    print("\n" + "="*60)
    print("TRANSFER LEARNING")
    print("="*60)

    print("\nLoading datasets...")
    source_data = ContactDataset(
        'datasets/contact-primary-school', 'contact-primary-school')
    target_data = ContactDataset(
        'datasets/contact-high-school', 'contact-high-school')

    # Transfer leraning from the primary school dataset to the high school dataset
    # I selected the primary school dataset as our source as it has fewer nodes but more dense structure.
    source_train_mask, source_val_mask, source_test_mask = DataSplitter.create_transductive_split(
        source_data.labels)
    target_train_mask, target_val_mask, target_test_mask = DataSplitter.create_transductive_split(
        target_data.labels)

    print(f"\n{'='*20} PHASE 1: PRE-TRAINING ON PRIMARY SCHOOL {'='*20}")

    hyperparams = {
        "input_dim": source_data.num_nodes,
        "hidden_dim": 16,
        "num_layers": 3,
        "alpha": 0.05,  # Lower alpha for denser primary school network
        "dropout": 0.1
    }

    with mlflow.start_run(run_name="Pretraining on Primary School"):
        mlflow.log_params(hyperparams)

        source_model = HypergraphGRAND(
            input_dim=hyperparams["input_dim"],
            hidden_dim=hyperparams["hidden_dim"],
            num_layers=hyperparams["num_layers"],
            alpha=hyperparams["alpha"],
            dropout=hyperparams["dropout"]
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        source_trainer = HypergraphTrainer(source_model, device)

        source_optimizer = torch.optim.Adam(
            source_model.parameters(), lr=0.01, weight_decay=1e-5)

        source_results = source_trainer.train(
            source_data,
            source_train_mask,
            source_val_mask,
            source_optimizer,
            num_epochs=10
        )

        source_test_results = source_trainer.evaluate(
            source_data, source_test_mask)

        print("\nSource Domain Results:")
        print(
            f"  - Best Val Accuracy: {source_results['best_val_accuracy']:.4f}")
        print(f"  - Test Accuracy: {source_test_results['test_accuracy']:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("best_val_accuracy",
                          source_results['best_val_accuracy'])
        mlflow.log_metric(
            "test_accuracy", source_test_results['test_accuracy'])
        mlflow.log_metric("test_loss", source_test_results['test_loss'])

        mlflow.pytorch.log_model(source_model, artifact_path="model")

    print(f"\n{'='*20} PHASE 2: TRANSFER TO HIGH SCHOOL {'='*20}")

    with mlflow.start_run(run_name="Transfer Learning to High School"):
        target_model = HypergraphGRAND(
            input_dim=target_data.num_nodes,
            hidden_dim=hyperparams["hidden_dim"],
            num_layers=hyperparams["num_layers"],
            # Higher alpha value for sparser high school network
            alpha=hyperparams["alpha"] * 2,
            dropout=hyperparams["dropout"]
        )

        # Transfer weights from source model (except input layer)
        target_state_dict = target_model.state_dict()
        source_state_dict = source_model.state_dict()

        # Transfer all weights except input_transform (different dimensions 327 vs 242)
        for name, param in source_state_dict.items():
            if name in target_state_dict and 'input_transform' not in name:
                target_state_dict[name].copy_(param)

        print("Transferred weights from source model (except input layer)")

        # Fine-tuning with different optimizers
        target_trainer = HypergraphTrainer(target_model, device)

        # Fine-tuning with SGD (lower learning rate)
        print("\nFine-tuning with SGD optimizer...")
        target_optimizer = torch.optim.SGD(
            target_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        target_results = target_trainer.train(
            target_data,
            target_train_mask,
            target_val_mask,
            target_optimizer,
            num_epochs=100
        )

        target_test_results = target_trainer.evaluate(
            target_data, target_test_mask)

        print("\nTarget Domain Results (Transfer Learning):")
        print(
            f"  - Best Val Accuracy: {target_results['best_val_accuracy']:.4f}")
        print(f"  - Test Accuracy: {target_test_results['test_accuracy']:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("best_val_accuracy",
                          target_results['best_val_accuracy'])
        mlflow.log_metric(
            "test_accuracy", target_test_results['test_accuracy'])
        mlflow.log_metric("test_loss", target_test_results['test_loss'])

        mlflow.pytorch.log_model(target_model, artifact_path="model")

    print(f"\n{'='*20} BASELINE: TRAIN FROM SCRATCH ON HIGH SCHOOL {'='*20}")

    with mlflow.start_run(run_name="Baseline Training on High School"):
        baseline_model = HypergraphGRAND(
            input_dim=target_data.num_nodes,
            hidden_dim=64,
            num_layers=3,
            alpha=0.1,
            dropout=0.1
        )

        baseline_trainer = HypergraphTrainer(baseline_model, device)
        baseline_optimizer = torch.optim.Adam(
            baseline_model.parameters(), lr=0.01, weight_decay=1e-5)

        baseline_results = baseline_trainer.train(
            target_data,
            target_train_mask,
            target_val_mask,
            baseline_optimizer,
            num_epochs=100
        )

        baseline_test_results = baseline_trainer.evaluate(
            target_data, target_test_mask)

        print("\nBaseline Results (From Scratch):")
        print(
            f"  - Best Val Accuracy: {baseline_results['best_val_accuracy']:.4f}")
        print(
            f"  - Test Accuracy: {baseline_test_results['test_accuracy']:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("best_val_accuracy",
                          baseline_results['best_val_accuracy'])
        mlflow.log_metric(
            "test_accuracy", baseline_test_results['test_accuracy'])
        mlflow.log_metric("test_loss", baseline_test_results['test_loss'])

        mlflow.pytorch.log_model(baseline_model, artifact_path="model")

    print(f"\n{'='*20} TRANSFER LEARNING COMPARISON {'='*20}")
    print(f"Transfer Learning Test Accuracy: {
          target_test_results['test_accuracy']:.4f}")
    print(f"From Scratch Test Accuracy: {
          baseline_test_results['test_accuracy']:.4f}")
    improvement = target_test_results['test_accuracy'] - \
        baseline_test_results['test_accuracy']
    print(f"Improvement: {improvement:+.4f}")

    mlflow.log_metric("accuracy_improvement", improvement)

    return {
        'source_results': source_results,
        'source_test_results': source_test_results,
        'target_results': target_results,
        'target_test_results': target_test_results,
        'baseline_results': baseline_results,
        'baseline_test_results': baseline_test_results,
        'improvement': improvement
    }
