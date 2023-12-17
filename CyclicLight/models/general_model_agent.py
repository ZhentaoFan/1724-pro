from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Activation, Embedding, Conv2D, concatenate, add,\
    multiply, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import copy


class GeneralAgent(NetworkAgent):

    def build_network(self):
        ins0 = Input(shape=(12, self.num_feat), name="input_total_features")
        ins1 = Input(shape=(8, ), name="input_cur_phase")

        #  embedding
        # [batch, 8] -> [batch, 8, 4] -> [batch, 2, 4, 4] -> [batch, 4, 4]
        cur_phase_emb = Activation('sigmoid')(Embedding(2, 4, input_length=8)(ins1))
        cur_phase_emb = Reshape((2, 4, 4))(cur_phase_emb)
        cur_phase_feat = Lambda(lambda x: K.sum(x, axis=1), name="feature_as_phase")(cur_phase_emb)
        
        # [batch, 12, n] -> [batch, 12, 32]
        feat_emb = Dense(32, activation="sigmoid")(ins0)

        # split according lanes
        lane_feat_s = tf.split(feat_emb, 12, axis=1)

        #  feature fusion for each phase
        MHA1 = MultiHeadAttention(4, 32, attention_axes=1)
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
        Sum1 = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))
        
        phase_feats_map_2 = []
        for i in range(self.num_phases):
            tmp_feat_1 = tf.concat([lane_feat_s[idx] for idx in self.phase_map[i]], axis=1)
            tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)
            tmp_feat_3 = Mean1(tmp_feat_2)
            phase_feats_map_2.append(tmp_feat_3)

        # embedding
        phase_feat_all = tf.concat(phase_feats_map_2, axis=1)
        phase_feat_all = concatenate([phase_feat_all, cur_phase_feat])

        att_encoding = MultiHeadAttention(4, 32, attention_axes=1)(phase_feat_all, phase_feat_all)
        hidden = Dense(64, activation="relu")(att_encoding)
        hidden = Dense(32, activation="relu")(hidden)
        hidden_flat = Flatten()(hidden)  # hidden is the output from the previous layer, now shape [None, 80]
        q_values = Dense(2, activation="linear")(hidden_flat)  # Now shape [None, 2]
        network = Model(inputs=[ins0, ins1],
                        outputs=q_values)
        
        network.compile()
        network.summary()
        return network
    
    def build_network2(self):
        ins0 = Input(shape=(12, self.num_feat), name="input_total_features")
        ins1 = Input(shape=(8, ), name="input_cur_phase")
        
        phase_feats = Flatten()(ins0)
        phase_feat_all = concatenate([phase_feats, ins1])
        hidden = Dense(64, activation="relu")(phase_feat_all)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Flatten()(hidden) 
        q_values = Dense(2, activation="linear")(hidden)

        network = Model(inputs=[ins0, ins1],
                        outputs=q_values)
        
        network.compile()
        network.summary()
        return network
    
    # CyclicLight's Action
    def choose_action(self, states):
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        # print(used_feature)
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
                    
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        q_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        new_actions = []
        for inter_id in range(self.num_intersections):
            if q_values[inter_id][0] > q_values[inter_id][1]:  # If 'keep' action has higher Q-value
                new_actions.append(self.last_actions[inter_id])  # Keep the same action
            else:
                # Change to the next cyclic action
                new_actions.append((self.last_actions[inter_id] + 1) % self.num_phases)

        self.last_actions = new_actions  # Update the last actions
        return new_actions    

    def choose_action2(self, states):
        # for cycle control
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
                    
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        q_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        action = np.argmax(q_values, axis=1)
        
        # based on 
        c_action = np.copy(action)
        for inter_id in range(self.num_intersections):
            
            if action[inter_id] == self.cyclicInd2[inter_id]:
                pass
            else:
                self.cyclicInd2[inter_id] = (self.cyclicInd2[inter_id] + 1) % self.num_phases
                c_action[inter_id] = self.cyclicInd2[inter_id]
            
        return c_action 

    # deactivation
    def choose_action4(self, states):
        # for cycle control
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
                    
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        q_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        action = np.argmax(q_values, axis=1)
        
        q_values_copy = q_values.copy()

        row_indices = np.arange(q_values.shape[0])
        # Set the largest values to -infinity
        q_values_copy[row_indices, action] = -np.inf

        # Get the second largest values
        second_arg_action = np.argmax(q_values_copy, axis=1)
        
        # deactivating the action that has been taken for too long
        c_action = np.copy(action)
        for inter_id in range(self.num_intersections):
            
            if(self.actionHisto[inter_id][action[inter_id]] > 0):
                self.actionHisto[inter_id][action[inter_id]] += 1
                if(self.actionHisto[inter_id][action[inter_id]] > self.maxActionTime):
                    self.actionHisto[inter_id] = np.zeros(self.num_phases)
                    self.actionHisto[inter_id][second_arg_action[inter_id]] = 1
                    c_action[inter_id] = second_arg_action[inter_id]    
            else:
                # clean action histo
                self.actionHisto[inter_id] = np.zeros(self.num_phases)
                self.actionHisto[inter_id][action[inter_id]] = 1
                c_action[inter_id] = action[inter_id]
                
        return c_action


    # Activation
    def choose_action5(self, states):
        # for cycle control
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
                    
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        q_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        action = np.argmax(q_values, axis=1)
        print(q_values)
        # activating the action that has been waiting for too long
        c_action = np.copy(action)
        
        # update action histo by adding 1 to all actions
        self.actionHisto = [[e+1 for e in row] for row in self.actionHisto]
        
        for inter_id in range(self.num_intersections):
            if(np.max(self.actionHisto[inter_id]) > self.waitingThreshold):
                c_action[inter_id] = np.argmax(self.actionHisto[inter_id])
                self.actionHisto[inter_id][c_action[inter_id]] = 0
            else:
                self.actionHisto[inter_id][action[inter_id]] = 0
        return c_action
    

    # Activation for side walk
    def choose_action6(self, states):
        # for cycle control
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
                    
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        q_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        action = np.argmax(q_values, axis=1)
        # print(q_values)
        # activating the action that has been waiting for too long
        c_action = np.copy(action)
        
        # update action histo by adding 1 to all actions
        self.actionHisto = [[e+1 for e in row] for row in self.actionHisto]
        
        for inter_id in range(self.num_intersections):
            if(self.actionHisto[inter_id][1] > self.pedestrianThreshold):
                c_action[inter_id] = 1
                self.actionHisto[inter_id][c_action[inter_id]] = 0
            elif (self.actionHisto[inter_id][3] > self.pedestrianThreshold):
                c_action[inter_id] = 3
                self.actionHisto[inter_id][c_action[inter_id]] = 0
            else:
                self.actionHisto[inter_id][action[inter_id]] = 0
        return c_action

    def prepare_samples(self, memory):
        """
        [state, action, next_state, final_reward, average_reward]
        """
        state, action, next_state, p_reward, ql_reward = memory
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        
        memory_size = len(action)
        
        _state = [[], None]
        _next_state = [[], None]
        for feat_name in used_feature:
            if feat_name == "new_phase":
                _state[1] = np.array(state[feat_name])
                _next_state[1] = np.array(next_state[feat_name])
            else:
                _state[0].append(np.array(state[feat_name]).reshape(memory_size, 12, -1))
                _next_state[0].append(np.array(next_state[feat_name]).reshape(memory_size, 12, -1))
                
        # ========= generate reaward information ===============
        if "pressure" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys():
            my_reward = p_reward
        elif "queue_length" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys() :
            my_reward = ql_reward
        else:
            my_reward = list(- (np.sum(next_state["lane_run_in_part"], axis=-1) + np.sum(next_state["lane_queue_in_part"], axis=-1)*self.dic_traffic_env_conf["RW"] )/4)
        
        return [np.concatenate(_state[0], axis=-1), _state[1]], action, [np.concatenate(_next_state[0], axis=-1), _next_state[1]], my_reward


    def train_network(self, memory):
        _state, _action, _next_state, _reward = self.prepare_samples(memory)
        
        # ==== shuffle the samples ============ 
        percent = self.dic_traffic_env_conf["PER"]
        # np.random.seed(int(percent*100))
        
        random_index = np.random.permutation(len(_action))
        _state[0] = _state[0][random_index, :, :]
        _state[1] = _state[1][random_index, :]
        _action = np.array(_action)[random_index]
        _next_state[0] = _next_state[0][random_index, :, :]
        _next_state[1] = _next_state[1][random_index, :]
        _reward = np.array(_reward)[random_index]

        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        num_batch = int(np.floor((len(_action) / batch_size)))

        loss_fn = MeanSquaredError()
        optimizer = Adam(lr=self.dic_agent_conf["LEARNING_RATE"])

        for epoch in range(epochs):

            for ba in range(int(num_batch*percent)):
                # prepare batch data
                batch_Xs1 = [_state[0][ba*batch_size:(ba+1)*batch_size, :, :], _state[1][ba*batch_size:(ba+1)*batch_size, :]]   
                batch_Xs2 = [_next_state[0][ba*batch_size:(ba+1)*batch_size, :, :], _next_state[1][ba*batch_size:(ba+1)*batch_size, :]]
                batch_r = _reward[ba*batch_size:(ba+1)*batch_size]
                batch_a = _action[ba*batch_size:(ba+1)*batch_size]

                cyclic_batch_Xs1 = [[],[]]
                cyclic_batch_Xs2 = [[],[]]
                cyclic_batch_r = []
                cyclic_batch_a = []
                for i in range(batch_size):
                    # cyclic action
                    if self.isPhaseCyclic(batch_Xs1[1][i,:], batch_Xs2[1][i,:]):
                        cyclic_batch_a.append(1)
                        cyclic_batch_r.append(batch_r[i])
                        cyclic_batch_Xs1[0].append(batch_Xs1[0][i,:,:])
                        cyclic_batch_Xs1[1].append(batch_Xs1[1][i,:])
                        cyclic_batch_Xs2[0].append(batch_Xs2[0][i,:,:])
                        cyclic_batch_Xs2[1].append(batch_Xs2[1][i,:])

                    # same action                    
                    if np.array_equal(batch_Xs1[1][i,:], batch_Xs2[1][i,:]):
                        cyclic_batch_a.append(0)
                        cyclic_batch_r.append(batch_r[i])
                        cyclic_batch_Xs1[0].append(batch_Xs1[0][i,:,:])
                        cyclic_batch_Xs1[1].append(batch_Xs1[1][i,:])
                        cyclic_batch_Xs2[0].append(batch_Xs2[0][i,:,:])
                        cyclic_batch_Xs2[1].append(batch_Xs2[1][i,:])
                        
                if len(cyclic_batch_Xs1[0]) > 0 and len(cyclic_batch_Xs1[1]) > 0:
                    batch_Xs1 = [np.array(cyclic_batch_Xs1[0]), np.array(cyclic_batch_Xs1[1])]
                    batch_Xs2 = [np.array(cyclic_batch_Xs2[0]), np.array(cyclic_batch_Xs2[1])]
                    batch_r = np.array(cyclic_batch_r)
                    batch_a = np.array(cyclic_batch_a)
                    # forward
                    with tf.GradientTape() as tape:
                        tape.watch(self.q_network.trainable_weights)
                        # calcualte basic loss
                        tmp_cur_q = self.q_network(batch_Xs1)
                        tmp_next_q = self.q_network_bar(batch_Xs2)
                        tmp_target = np.copy(tmp_cur_q)
                        for i in range(len(cyclic_batch_Xs1[0])):
                            tmp_target[i, batch_a[i]] = batch_r[i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                        self.dic_agent_conf["GAMMA"] * \
                                                        np.max(tmp_next_q[i, :])
                        base_loss = tf.reduce_mean(loss_fn(tmp_target, tmp_cur_q)) #?
                                        
                        # calculate CQL loss
                        replay_action_one_hot = tf.one_hot(batch_a, 2, 1., 0., name='action_one_hot')
                        replay_chosen_q = tf.reduce_sum(tmp_cur_q * replay_action_one_hot)
                        dataset_expec = tf.reduce_mean(replay_chosen_q) 
                        negative_sampling = tf.reduce_mean(tf.reduce_logsumexp(tmp_cur_q, 1))
                        min_q_loss = (negative_sampling - dataset_expec)
                        min_q_loss = min_q_loss * self.min_q_weight
                        
                        print('min_q_loss: ', min_q_loss)
                        print('base_loss: ', base_loss)
                        
                        tmp_loss = base_loss + min_q_loss
                        
                        # # Assume `batch_a` is a tensor of shape (batch_size,) with integer action values
                        # # And `num_actions` is the total number of possible actions
                        # num_actions = 4
                        # batch_size = tf.size(batch_a)

                        # # Count the occurrence of each action to estimate the probability distribution of `batch_a`
                        # action_counts = tf.math.bincount(batch_a, minlength=num_actions)
                        # action_probs = action_counts / tf.reduce_sum(action_counts)
                        # action_probs = tf.cast(action_probs, tf.float32)
                        # # Define the prior distribution
                        # prior_probs = tf.constant([0.25, 0.25, 0.25, 0.25])  # prior is uniform
                        

                        # # Add a small value to probabilities to prevent log(0)
                        # epsilon = 1e-8
                        # action_probs += epsilon
                        # prior_probs += epsilon

                        # # Normalize the probabilities to ensure they sum up to 1
                        # action_probs /= tf.reduce_sum(action_probs)
                        # prior_probs /= tf.reduce_sum(prior_probs)

                        # # Calculate KL divergence
                        # kl_divergence = tf.reduce_sum(action_probs * tf.math.log(action_probs / prior_probs))

                        # # print("baseloss datatype: ", base_loss.dtype)
                        # # print("min_q_loss datatype: ", min_q_loss.dtype)
                        # # print("KL loss datatype:", kl_divergence.dtype)
                        # # print("weighted KL loss datatype:", self.kl_divergence_weight * kl_divergence.dtype)
                        # # Add the KL divergence as a regularizer to your loss function
                        # tmp_loss = base_loss + min_q_loss + self.kl_divergence_weight * kl_divergence
                        
                        # print("baseloss: ", base_loss)
                        # print("min_q_loss: ", min_q_loss)
                        # print("KL loss:", kl_divergence)
                        # print("weighted KL loss:", self.kl_divergence_weight * kl_divergence)
                    

                        grads = tape.gradient(tmp_loss, self.q_network.trainable_weights)
                        optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))
                    
                    
    def isCyclic(self, prevAction, curAction):
        if prevAction == curAction:
            return True
        elif (prevAction + 1) % self.num_phases == curAction:
            return True
        else:
            return False
        

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
            
        