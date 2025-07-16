# Hypergraph GRAND

Based on GRAND: Graph Neural Diffusion. TODO: Add the proper citation and refs.

## Installation and running
If you see a `venv/` folder please delete it, then just run `make` if you are in a unix based system.

For Windows, you can manually create a virtual environment, source it, use `pip install -r requirements.txt` then run `python main.py` and it should start training the model. There are two current approaches for two different datasets:

- Transductive learning approach: You can find the details in `approaches/transductive.py`
- Transfer learning approach: You can find the details in `approaches/transfer.py`

Contact: maharjaa@myumanitoba.ca
