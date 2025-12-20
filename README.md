# Hierarchical Federated Learning with P2P vs SHM+P2P Communication

## Project Overview

This project evaluates how different communication backends impact the performance of **hierarchical federated learning (FL)** using the **FLAME framework**. We compare two communication designs:

- **P2P only**: All communication is performed using point-to-point gRPC over the network.
- **SHM + P2P**: Shared Memory (SHM) is used for intra-node communication, while P2P gRPC is used for inter-node communication.

The goal is to understand whether avoiding the network stack for processes running on the same machine can reduce communication overhead, improve runtime, and increase overall training efficiency.

The experiments are conducted using:
- **FEMNIST dataset**
- **ResNet-18 model**
- **Hierarchical FL topology** (Trainer → Leaf Aggregator → Middle Aggregator → Top Aggregator → Coordinator)
- **10 nodes total**, with 9 worker nodes and 1 coordinator/top-aggregator node

---

## Repository Structure



This repo has the following directory structure:
```
flame
 ├── CODE_OF_CONDUCT.md
 ├── CONTRIBUTING.md
 ├── LICENSE
 ├── Makefile -> build/Makefile
 ├── README.md
 ├── api (specification of REST API for flame apiserver)
 ├── build (configuration files for building flame binaries and container image)
 ├── cmd (source files for flame control plane)
 ├── docs (document folder)
 ├── examples (example folder)
 ├── fiab (dev/test env in a single box)
 ├── go.mod
 ├── go.sum
 ├── lib (python library for core flame data plane)
 ├── lint.sh
 ├── pkg (go packages for cmd)
 └── scripts (utility scripts)
```

## System Architecture

The experiments are conducted on a 10-node cluster with the following setup:

- **1 node**:
  - Coordinator
  - Top Aggregator

- **9 worker nodes** (each node runs):
  - 1 Middle Aggregator
  - 3 Leaf Aggregators
  - 6 Trainers

The hierarchical FL workflow follows:
Trainer → Leaf Aggregator → Middle Aggregator → Top Aggregator → Coordinator

All experiments use the FEMNIST dataset and a ResNet-18 model.


## Environment Setup

Run the following commands on each node before running the experiments.

```bash
# System packages
sudo apt update && sudo apt install -y byobu htop

# Storage setup
sudo chown -R $(id -u):$(id -g) /mydata
cd /mydata
export MYMOUNT=/mydata

# Install Go
golang_file=go1.22.3.linux-amd64.tar.gz
curl -LO https://go.dev/dl/$golang_file
tar -C /mydata -xzf $golang_file

# Update PATH
echo 'PATH="/mydata/go/bin:$PATH"' >> $HOME/.bashrc
echo 'PATH="$HOME/.flame/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc

# Install golangci-lint
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | \
sh -s -- -b /mydata/go/bin v1.49.0
golangci-lint --version

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /mydata/miniconda3

# Initialize conda
source /mydata/miniconda3/bin/activate
conda init bash
source $HOME/.bashrc

# Create and activate environment
conda create -n flame python=3.9 -y
conda activate flame

# Python dependencies
pip install google tensorflow torch torchvision

#for dataset
pip install google tensorflow torch torchvision mlflow "flwr-datasets[vision]"

git clone -b p2p-vs-mixed-backend https://github.com/Sumit-Haldar1/lifl_flame.git

#Rename it to flame

cd /mydata/flame # && git checkout duplicate-clients
make install # Install flame control plane utilities
cd lib/python && make install # Install flame's py environment

#SHM + eBPF Backend Setup

# 1. Install dependencies for libbpf
sudo apt update && sudo apt install -y \
  flex bison build-essential dwarves libssl-dev \
  libelf-dev pkg-config libconfig-dev clang gcc-multilib

# 2. Build and install libbpf (version 0.6.0)
cd /mydata/flame/third_party/spright_utility/scripts
./libbpf.sh

# 3. Fix LIBBPF_0.6.0 runtime error by linking correct library
cd /mydata/flame/third_party/spright_utility/scripts/libbpf/src
sudo cp libbpf.so.0.6.0 /lib/x86_64-linux-gnu/
sudo ln -sf /lib/x86_64-linux-gnu/libbpf.so.0.6.0 /lib/x86_64-linux-gnu/libbpf.so.0
sudo ldconfig

# 4. Compile the sockmap_manager binary
cd /mydata/flame/third_party/spright_utility/src
gcc -o sockmap_manager sockmap_manager.c -lbpf -lelf
mkdir -p ../bin
mv sockmap_manager ../bin/

# To Run metaserver 
cd ~/
sudo .flame/bin/metaserver


```

## Run the Experiments (P2P-only vs SHM+P2P)

### 1) Go to the ResNet18 example directory

```bash
cd /mydata/flame/lib/python/examples/resnet18
pwd
ls
```

You will see **5 role folders**:

- `coordinator/`
- `top_aggregator/`
- `middle_aggregator/`
- `leaf_aggregator/`
- `trainer/`

Each folder contains a `main.py` and **two config files** (one for **P2P-only**, one for **SHM+P2P hybrid**).

---

### 2) Pick the mode (choose ONE)

#### Mode A: P2P-only
Use the config file that has **p2p** backend everywhere (example name pattern: `config_p2p.json`).

#### Mode B: Hybrid (SHM + P2P)
Use the config file that uses:
- **shm** for **intra-node channels**
- **p2p** for **inter-node channels**
(example name pattern: `config_shm_p2p.json`)

> You will run the **same roles**, only the **config file changes**.

---

### 3) Start processes in this order

#### (A) Coordinator (on the node that hosts Coordinator + Top Aggregator)
```bash
cd /mydata/flame/lib/python/examples/resnet18/coordinator
python main.py <CONFIG_FILE>
```

#### (B) Top Aggregator (same node as Coordinator)
```bash
cd /mydata/flame/lib/python/examples/resnet18/top_aggregator
python main.py <CONFIG_FILE>
```

#### (C) Middle Aggregators (worker nodes)
```bash
cd /mydata/flame/lib/python/examples/resnet18/middle_aggregator
python main.py <CONFIG_FILE>
```

#### (D) Leaf Aggregators (worker nodes)
```bash
cd /mydata/flame/lib/python/examples/resnet18/leaf_aggregator
python main.py <CONFIG_FILE>
```

#### (E) Trainers (worker nodes)
```bash
cd /mydata/flame/lib/python/examples/resnet18/trainer
python main.py <CONFIG_FILE>
```

---

### 4) Replace `<CONFIG_FILE>` correctly

Examples (use whatever your repo actually names them):

- **P2P-only run**
```bash
python main.py config_p2p.json
```

- **Hybrid run (SHM + P2P)**
```bash
python main.py config_shm_p2p.json
```

---

### 5) Logs for analysis

After the run finishes, collect logs from **Top Aggregator** (and optionally Coordinator) for:
- accuracy vs round/time
- round duration
- CPU time/utilization
- communication time breakdown


