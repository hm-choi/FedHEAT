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
The library is provided by CryptoLab as a Docker image via https://hub.docker.com/r/cryptolabinc/heaan-stat, and all experiments were conducted in containers created from the '1.0.0-gpu' tagged image.

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

## 3. Evaluation results
We compare the server-side aggregation time (in seconds) of FedHEAT, FedHEAT without MR (Moment Reparameterization, our proposed method), and HE-Adam [2] across five benchmark datasets.
For a fair comparison across datasets, we measure the aggregation time over 500 rounds for each dataset.

### Table. Server-side aggregation runtime (seconds) under different choices of the stabilization parameter τ across five benchmarks. The experiments for this table were conducted on Server1.
| Setting    | Method             | FEMNIST  | CelebA | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-------------|--------------------|----------|---------|-----------|-------------|----------------|
| Ciphertext | HE-Adam            | 14020.47 | 96.99   | 23114.19  | 23372.54    | 23481.53       |
| Ciphertext | FedHEAT (w/o MR)   | 10755.49 | 82.95   | 17915.45  | 18034.93    | 18108.73       |
| Ciphertext | **FedHEAT**        | 9171.14  | 73.78   | 14992.82  | 14898.50    | 15055.01       |

We compare the teste accuracy of FedHEAT with FedAdam[4] in the plaintext setting, and FedHEAT with its variant and HE-Adam in the ciphertext setting.
### Table. Accuracy (%) of FedHEAT variants and HE-Adam across benchmark datasets. The experiments for this table were conducted on Server1 and Server2. Server1 was used for all ciphertext-setting experiments except CIFAR-100, while Server2 was used for the plaintext-setting experiments and the ciphertext-setting experiment on CIFAR-100.

| Setting    | Method             | FEMNIST | CelebA | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-------------|--------------------|----------|---------|-----------|-------------|----------------|
| Plaintext  | FedAdam            | 84.85    | 89.07   | 85.52     | 55.73       | 41.54          |
| Plaintext  | FedHEAT            | 84.91    | 90.84   | 87.86     | 55.16       | 42.05          |
| Ciphertext | HE-Adam            | 84.82    | 90.70   | 87.46     | 55.23       | 43.09          |
| Ciphertext | FedHEAT (w/o MR)   | 84.77    | 90.60   | 86.70     | 54.54       | 41.68          |
| Ciphertext | **FedHEAT**        | 84.79    | 90.91   | 87.66     | 55.26       | 41.80          |

## 4. License
This is available for non-commercial purposes only.

## 5. References
[1] Geeho Kim, Jinkyu Kim, and Bohyung Han. Communication-efficient federated learning with accelerated client gradient. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12385–12394, 2024. \
[2] Chi-Hieu Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Kristin Lauter, and Miran Kim. Empowering artificial intelligence with homomorphic encryption for secure deep reinforcement learning. Nature Machine Intelligence, pages 1–14, 2025. \
[3] CryptoLab. HEaaN Library. https://www.cryptolab.co.kr/en/products-en/heaan-he/, 2022. Accessed: 2026-04-24. \
[4] Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneˇcný, Sanjiv Kumar, and H. Brendan McMahan. Adaptive Federated Optimization. In International Conference on Learning Representations, 2021.