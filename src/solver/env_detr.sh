conda install -n detr pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
#above install many packages, above install may lead libtorch_cpu.so: undefined symbol: iJIT_IsProfilingActive
# https://pytorch.org/get-started/previous-versions/
The following packages are incompatible
├─ python 3.12.5  is requested and can be installed;
├─ pytorch-cuda 11.8**  is requested and can be installed;
└─ torchaudio 2.0.2  is not installable because there are no viable options
   ├─ torchaudio 2.0.2 would require
   │  └─ python >=3.8,<3.9.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchaudio 2.0.2 would require
   │  └─ python >=3.11,<3.12.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchaudio 2.0.2 would require
   │  └─ python >=3.10,<3.11.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchaudio 2.0.2 would require
   │  └─ pytorch-cuda 11.7.* , which conflicts with any installable versions previously reported;
   └─ torchaudio 2.0.2 would require
      └─ python >=3.9,<3.10.0a0 , which conflicts with any installable versions previously reported.
#conad install python=3.9 -c conda-forge
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip uninstall libpng libjpeg


pip install pandas
pip install pycocotools
pip install PyYAML
pip install scipy
pip install tqdm
pip install rdkit
conda install conda-forge::xorg-libxrender
conda install conda-forge::xorg-libxext
pip install  scikit-learn
pip install scikit-image
#for rdkit draw ImportError: libXrender.so.1: cannot open shared object file: No such file or directory
pip install ipykernel
