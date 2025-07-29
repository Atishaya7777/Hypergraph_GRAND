import torch
import numpy as np
import json

from approaches.transductive import transductive_learning_approach


def main():
    print("HyperGRAND: Hypergraph Graph Neural Diffusion")
    print("Clustering-based learning on contact network datasets")
    print("="*60)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        transductive_results = transductive_learning_approach(dataset_name='planetoid_cora',strategy='classification')
        # transfer_results = transfer_learning_approach()

        results_summary = {
            'transductive_results': transductive_results,
            # 'transfer_results': transfer_results,
            'timestamp': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
        }

        with open('hypergrand_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
