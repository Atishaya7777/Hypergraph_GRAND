from data.dataset import create_hypergraph_dataset

planetoid_cora = create_hypergraph_dataset('planetoid_cora', hypergraph_strategy='neighborhood_expansion')
data = planetoid_cora.load_data('/tmp/cora_data')

print(f"Cora dataset statistics: {data.dataset_info}")
