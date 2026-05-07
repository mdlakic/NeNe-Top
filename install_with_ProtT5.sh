conda create -n NeNe -c conda-forge python=3.8 -y
conda activate NeNe
pip install bio-embeddings
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install 'transformers[torch]'
pip install sentencepiece
pip install tqdm==4.64.1
pip install biopython==1.81
pip install joblib==1.1.0
pip install numpy==1.22.3
pip install scikit_learn==1.1.0
pip install pandas==1.5.3
pip install tensorflow==2.8.0
pip install keras==2.8.0
pip install protobuf==3.20.1
