In order to create our environment:

1) Create a Conda environment:
    conda env create -f environment.yml

    conda activate ProjectENV

2) Install some pip libraries independently:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    pip install torch-fidelity torchmetrics

Afterwards, run the python files simply with: 
    python <file_name>