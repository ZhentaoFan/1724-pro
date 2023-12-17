## Structure 

**./generate_offline_data** is the directory to generate offline data based for different agents and maps.

**./data** includes the roadnet and the traffic info data.

**./model** includes the Q-network trained with different configurations.

**./models** includes the different agent models. The network for CyclicLight agent is build_network() in general_model_agent.py.

**./records** is where the training and testing results stored.

## Training & Testing
```
python run_offline.py
```
