#!/usr/bin/env python3
"""
HyperGRAND: Hypergraph Graph Neural Diffusion
Main entry point with command-line interface for training and evaluation.
Supports both Planetoid datasets and custom contact network datasets.
Supports both clustering and classification strategies.
"""

import argparse
import json
import torch
import numpy as np
from typing import List

from approaches.transductive import transductive_learning_approach


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HyperGRAND: Hypergraph Graph Neural Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            # Planetoid (Classification)
            "planetoid",
            "planetoid_cora",
            "planetoid_citeseer",
            "planetoid_pubmed",
            # Contact (Clustering)
            "contact",
            "contact_high_school",
            "contact_primary_school",
            # Clustering
            "walmart_trips",
            "stackoverflow_answers",
            "amazon_reviews",
            # Partitioning
            "zoo",
            "mushroom",
            "ntu2012",
            "modelnet40",
            "house_committees",
            # Other
            "20newsW100",
            "coauthorship",
            "cocitation",
            "yelp",
        ],
        default="planetoid_cora",
        help='Dataset to use. Can be Planetoid (classification), Contact networks (clustering), '
        'or other hypergraph datasets. See DATASET_METADATA.json for full list.',
    )

    # Task strategy
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["clustering", "classification"],
        default="classification",
        help="Learning strategy: clustering or classification",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden dimension of the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Diffusion parameter alpha"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["explicit", "implicit", "multistep", "adaptive"],
        default="explicit",
        help="Integration scheme for the diffusion process",
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs", type=int, default=5000, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=1000, help="Early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer"
    )

    # Edge dropout
    parser.add_argument(
        "--edge_dropout_rates",
        type=float,
        nargs="+",
        default=[0.75, 0.6, 0.5],
        help="Edge dropout rates to test (can specify multiple)",
    )

    # MLflow and logging
    parser.add_argument(
        "--log_detailed_params",
        action="store_true",
        default=True,
        help="Log detailed parameters to MLflow",
    )
    parser.add_argument(
        "--no_mlflow", action="store_true", help="Disable MLflow logging"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="hypergrand_results.json",
        help="Output JSON file for results",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("HyperGRAND: Hypergraph Graph Neural Diffusion")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print(f"Edge dropout rates: {args.edge_dropout_rates}")
    print(
        f"Model: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
        f"alpha={args.alpha}, dropout={args.dropout}, scheme={args.scheme}"
    )
    print(
        f"Training: epochs={args.num_epochs}, patience={args.patience}, "
        f"lr={args.learning_rate}, weight_decay={args.weight_decay}"
    )
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        # Handle "planetoid" case - run all planetoid datasets
        if args.dataset == "planetoid":
            # Run all planetoid datasets
            all_results = {}
            for dataset in ["planetoid_cora", "planetoid_citeseer", "planetoid_pubmed"]:
                print(f"\n{'='*60}")
                print(f"Running {dataset}")
                print(f"{'='*60}")
                dataset_results = transductive_learning_approach(
                    dataset_name=dataset,
                    strategy=args.strategy,
                    edge_dropout_rates=args.edge_dropout_rates,
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    log_detailed_params=args.log_detailed_params,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    alpha=args.alpha,
                    dropout=args.dropout,
                    scheme=args.scheme,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                all_results.update(dataset_results)
            results = all_results
        else:
            # Run single dataset
            results = transductive_learning_approach(
                dataset_name=args.dataset,
                strategy=args.strategy,
                edge_dropout_rates=args.edge_dropout_rates,
                num_epochs=args.num_epochs,
                patience=args.patience,
                log_detailed_params=args.log_detailed_params,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                alpha=args.alpha,
                dropout=args.dropout,
                scheme=args.scheme,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        # Save results
        results_summary = {
            "args": vars(args),
            "transductive_results": results,
            "timestamp": str(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
        }

        with open(args.output, "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        print(f"\nResults saved to {args.output}")
        print("\nSummary:")
        for key, result_data in results.items():
            if isinstance(result_data, dict) and "test_results" in result_data:
                test_acc = result_data["test_results"].get("test_accuracy", "N/A")
                val_acc = result_data["train_results"].get("best_val_accuracy", "N/A")
                print(
                    f"  {key}: Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}"
                    if isinstance(val_acc, (int, float))
                    and isinstance(test_acc, (int, float))
                    else f"  {key}: {result_data}"
                )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
