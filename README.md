# FedHEAT: Adaptive Optimization for Privacy-Preserving Federated Learning under Homomorphic Encryption

Official repository for **FedHEAT: Adaptive Optimization for Privacy-Preserving Federated Learning under Homomorphic Encryption**  

FedHEAT is an experimental toolkit for privacy-preserving federated learning under homomorphic encryption, built on the Adam optimizer, a representative adaptive optimization method. This toolkit is evaluated on the following five federated learning benchmarks:

- FEMNIST
- CelebA
- CIFAR-10
- CIFAR-100
- Tiny-ImageNet

This implementation is based on the FedACG[1] repository(https://github.com/geehokim/FedACG). All plaintext training pipelines and data preprocessing components are adapted from the original FedACG codebase.

## Abstract
Federated learning(FL) keeps raw data on clients, yet shared model updates can still leak sensitive information. Homomorphic Encryption (HE) addresses this risk by enabling aggregation directly over encrypted updates, but its computational cost has largely confined HE-based FL to simple averaging. Adaptive optimization on the server, which is critical under heterogenous client data, is particularly costly under HE because methods such as FedAdam rely on recursive moment updates and non-linear scaling that consume multiplicative depth and trigger frequent bootstrapping, the dominant cost in HE.

We propose FedHEAT, a framework for adaptive federated learning optimization under HE. FedHEAT reparameterizes the first- and second-moment states of FedAdam to eliminate per-round level consumption in encrypted moment updates, reducing bootstrapping frequency by up to two orders of magnitude. For the parameter update, FedHEAT halves the multiplicative depth of the encrypted inverse-square-root operation by bounding its input range. Across five non-IID FL benchmarks, FedHEAT achieves accuracy comparable to plaintext FedAdam, while reducing server runtime under HE by up to 36.3% over HE-Adam [2], a prior HE-compatible Adam optimizer.

## 1. Server Setting
- Experiments are conducted on two servers.
Server1 is equipped with an Intel Xeon w9-3475X CPU, an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU, and 192GB of RAM.
Server2 is equipped with an Intel Xeon 6521P CPU, an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU, and 384GB of RAM.
- The server used for each experiment is specified in the corresponding experiment description.
- We use the HEaaN-GPU library[3] with the CKKS scheme. 
The library is provided as a Docker image via https://hub.docker.com/r/cryptolabinc/heaan-statd, and all experiments were conducted in containers created from the '1.0.0-gpu' tagged image.

## 2. Run the codes
First, install the required Python packages listed in 'requirements.txt':
~~~bash
pip install -r ./requirements.txt
~~~
The experiments for FEMNIST and CelebA are implemented in the following directory:
./FedHEAT_leaf_dataset

The experiments for CIFAR-10/100, and Tiny-ImageNet are implemented in the following directory:
./FedHEAT_normal_dataset

Detailed explanations and how to run the code are provided in the README.md file in each directory.

## 3. License
This is available for non-commercial purposes only.

[1] Geeho Kim, Jinkyu Kim, and Bohyung Han. Communication-efficient federated learning with accelerated client gradient. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12385–12394, 2024.
[2] Chi-Hieu Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Kristin Lauter, and Miran Kim. Empowering artificial intelligence with homomorphic encryption for secure deep reinforcement learning. Nature Machine Intelligence, pages 1–14, 2025.
[3] CryptoLab. HEaaN Library. https://www.cryptolab.co.kr/en/products-en/heaan-he/, 2022. Accessed: 2026-04-24.