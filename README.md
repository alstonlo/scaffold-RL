# scaffold-RL
 
## Dependencies 

Please create and activate a conda environment with the dependencies listed in `environment.yml`. Note that 
there are some unused packages (from when I was playing around with GNNs using DGL). 

## File Structure 

The following summarizes the core files and folders of this repository 

- `src/`: the source code folder: 
  - `environments.py`: the MDP environment implementation  
  - `dqn.py`: the DQN implementation 
  - `agents.py`: the agent implementations (e.g. epsilon-greedy agent)
  - `experiments/`: contains the experimental scripts used in this project: 
    - `opt_qed_baselines.py`: script to sample molecules from greedy and epsilon-greedy agents
    - `train_qed_dqn.py`: script to train the DQN 
    - `opt_qed_dqn.py`: script to sample molecules under the DQN
  - `utils/`: contains various helper classes and functions used in throughout the project (e.g. a replay buffer 
  implementation, seeding utilites, and molecular graph manipulations)
- `results/`: contains the designed molecules from the various agents and trained model files 
  - `qed_baseline/`: contains the molecules designed from the epsilon-greedy agents
  - `qed_dqn/`: contains the molecules designed from the DQN

## Reproducability

All scripts are seeded and left in exactly the state they were run in. The experiments were performed on Google Colab or PyCharm. 
