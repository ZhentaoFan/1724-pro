import numpy as np
import os
from .agent import Agent
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def build_memory():
    return []


class CoLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.CNN_layers =  [[32, 32]]
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()
        self.last_actions = [0] * self.num_agents
        if cnt_round == 0:
            # initialization
            self.q_network = self.build_network()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)),
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                "UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "new_phase" in feat_name:
                N += 8
            else:
                N += 12
        return N

    @staticmethod
    def MLP(ins, layers=None):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,dim]
        -outpout: [batch,#agents,dim]
        """
        if layers is None:
            layers = [128, 128]
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(ins)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(h)

        return h

    def MultiHeadsAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        """
        input: [batch, agent, dim] feature
               [batch, agent, nei, agent] adjacency
        input:[bacth,agent,128]
        output:
              [batch, agent, dim]
        """
        # [batch,agent,dim]->[batch,agent,1,dim]
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)

        # [batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)

        # 获取neighbor的feature，包含自身，然后对这个序列做 self-attention
        # [batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr = Lambda(lambda x: tf.matmul(x[0], x[1]))([in_nei, neighbor_repr])

        # attention computation
        # [batch, agent, 1, dim]->[batch, agent, 1, h_dim*head]
        agent_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                name='agent_repr_%d' % suffix)(agent_repr)
        # [batch,agent,1,h_dim,head]->[batch,agent,head,1,h_dim]
        agent_repr_head = Reshape((self.num_agents, 1, h_dim, head))(agent_repr_head)
        agent_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(agent_repr_head)

        # [batch,agent,neighbor,dim]->[batch,agent,neighbor,h_dim*head]
        neighbor_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                   name='neighbor_repr_%d' % suffix)(neighbor_repr)
        # [batch,agent,neighbor,h_dim,head]->[batch,agent,head,neighbor,h_dim]
        neighbor_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(neighbor_repr_head)
        neighbor_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(neighbor_repr_head)

        # [batch,agent,head,1,h_dim]x[batch,agent,head,neighbor,h_dim]->[batch,agent,head,1,neighbor]
        att = Lambda(lambda x: K.softmax(tf.matmul(x[0], x[1], transpose_b=True)))([agent_repr_head,
                                                                                    neighbor_repr_head])
        # [batch,agent,head,1,neighbor]->[batch,agent,head,neighbor]
        att_record = Reshape((self.num_agents, head, self.num_neighbors))(att)

        # self embedding again, [batch, agent, nei, dim] -> [batch, agent, nei, h_dim*head ]
        neighbor_hidden_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                          name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        # [batch, agent, nei, h_dim, head]
        neighbor_hidden_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(
            neighbor_hidden_repr_head)
        # [batch, agent, head, nei, h_dim]
        neighbor_hidden_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(
            neighbor_hidden_repr_head)
        # [batch, agent, head, 1, nei] * [batch, agent, head, nei, h_dim] -> [batch, agent, head, 1, h_dim]
        out = Lambda(lambda x: K.mean(tf.matmul(x[0], x[1]), axis=2))([att, neighbor_hidden_repr_head])
        out = Reshape((self.num_agents, h_dim))(out)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal', name='MLP_after_relation_%d' % suffix)(
            out)
        return out, att_record

    def adjacency_index2matrix(self, adjacency_index):
        # [batch,agents,neighbors]
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        lab = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return lab

    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        # TODO
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        tmp.extend(self.dic_traffic_env_conf['PHASE'][s[i][feature][0]])
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])

            feats0.append(tmp)
        # [1, agent, dim]
        # feats = np.concatenate([np.array([feat1]), np.array([feat2])], axis=-1)
        feats = np.array([feats0])
        # [1, agent, nei, agent]
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj]

    def choose_action(self, states):
        """
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        """
        xs = self.convert_state_to_input(states)
        q_values = self.q_network(xs)
        # TODO: translate from cyclic to normal
        # if random.random() <= self.dic_agent_conf["EPSILON"]:
        #     action = np.random.randint(self.num_actions, size=len(q_values[0]))
        # else:
        action = np.argmax(q_values[0], axis=1)
        new_actions = []
        for inter_id in range(self.num_agents):
            if action[inter_id] == 0:  # If 'keep' action has higher Q-value
                new_actions.append(self.last_actions[inter_id])  # Keep the same action
            else:
                # Change to the next cyclic action
                new_actions.append((self.last_actions[inter_id] + 1) % self.num_actions)

        self.last_actions = new_actions  # Update the last actions
        
        return new_actions

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        """
        memory: [slice_data, slice_data, ..., slice_data]
        prepare memory for training
        """
        # ToDo: clean data
        slice_size = len(memory[0])
        _adjs = []
        # state : [feat1, feat2]
        # feati : [agent1, agent2, ..., agentn]
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        # print(len(memory))
        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = memory[j][i]
                # ToDo: clean data
                # print(state["new_phase"])
                # print(next_state["new_phase"])
                if self.isPhaseCyclic(np.array(state["new_phase"]),np.array(next_state["new_phase"])):
                    _action[j].append(1)
                    _reward[j].append(reward)
                    _adj.append(state["adjacency_matrix"])
                    print('cyclic')
                    _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                    _next_state[j].append(self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))
                elif np.array_equal(np.array(state["new_phase"]), np.array(next_state["new_phase"])):
                    _action[j].append(0)
                    _reward[j].append(reward)
                    _adj.append(state["adjacency_matrix"])
                    print('keep')
                    _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                    _next_state[j].append(self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))
                
            _adjs.append(_adj)
        # [batch, agent, nei, agent]
        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))

        # [batch, 1, dim] -> [batch, agent, dim]
        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1)
        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1)
        target = self.q_network([_state2, _adjs2])
        next_state_qvalues = self.q_network_bar([_next_state2, _adjs2])
        # [batch, agent, num_actions]
        final_target = np.copy(target)
        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, _action[j][i]] = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[i, j])

        self.Xs = [_state2, _adjs2]
        self.Y = final_target

    def build_network(self, MLP_layers=[32, 32]):
        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)
        In = list()
        # In: [batch,agent,dim]
        # In: [batch,agent,neighbors,agents]
        In.append(Input(shape=(self.num_agents, self.len_feature), name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))

        feature = self.MLP(In[0], MLP_layers)

        # feature:[batch,agents,feature_dim]
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index, CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:", CNN_heads[CNN_layer_index])
            if CNN_layer_index == 0:
                h, _ = self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    d_in=MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
            else:
                h, _ = self.MultiHeadsAttModel(
                    h,
                    In[1],
                    d_in=MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
        # action prediction layer
        # [batch,agent,32]->[batch,agent,action]
        out = Dense(2, kernel_initializer='random_normal', name='action_layer')(h)
        # out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model = Model(inputs=In, outputs=out)

        model.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                      loss=self.dic_agent_conf["LOSS_FUNCTION"])
        model.summary()
        return model

    def train_network(self, memory):
        self.prepare_Xs_Y(memory)
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                           verbose=2, validation_split=0.3, callbacks=[early_stopping])

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D})
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def isPhaseCyclic(self, prevPhase, curPhase):
        phase1 = np.asarray([0, 1, 0, 1, 0, 0, 0, 0])
        phase2 = np.asarray([0, 0, 0, 0, 0, 1, 0, 1])
        phase3 = np.asarray([1, 0, 1, 0, 0, 0, 0, 0])
        phase4 = np.asarray([0, 0, 0, 0, 1, 0, 1, 0])
        if (np.array_equal(prevPhase, phase1) and np.array_equal(curPhase, phase2)) or \
                (np.array_equal(prevPhase, phase2) and np.array_equal(curPhase, phase3)) or \
                (np.array_equal(prevPhase, phase3) and np.array_equal(curPhase, phase4)) or \
                    (np.array_equal(prevPhase, phase4) and np.array_equal(curPhase, phase1)):
            return True
        else:
            return False

class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        # [batch,agent,dim]->[batch,1,agent,dim]
        # [batch,1,agent,dim]->[batch,agent,agent,dim]
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

