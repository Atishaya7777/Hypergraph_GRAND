from approaches.transfer import transfer_learning_approach
from approaches.transductive import transductive_learning_approach
import torch
import numpy as np
import json


def main():
    print("HyperGRAND: Hypergraph Graph Neural Diffusion")
    print("Clustering-based learning on contact network datasets")
    print("="*60)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run both approaches
        transductive_results = transductive_learning_approach()
        transfer_results = transfer_learning_approach()

        # Print summary...
        # Save results to file
        results_summary = {
            'transductive_results': transductive_results,
            'transfer_results': transfer_results,
            'timestamp': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
        }

        with open('hypergrand_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
