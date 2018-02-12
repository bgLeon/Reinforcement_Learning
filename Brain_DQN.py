import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.set_random_seed(1)

class DQN_linear:

	def __init__ (self,n_features,n_actions,learning_rate=0.01,reward_decay=0.9,exploration=0.1,
	 replace_target_iterations=100, memory_size=500,batch_size=32,exploration_decrement=None,
	 max_score=500,output_graph=False):
		self.n_features=n_features
		self.n_actions=n_actions
		self.lr=learning_rate
		self.gamma=reward_decay
		self.epsilon_min=exploration
		self.rep_targ_iter=replace_target_iterations
		self.memory_size=memory_size
		self.max_score=max_score
		self.batch_size=batch_size
		self.exploration_decrement=exploration_decrement
		self.epsilon=1 if exploration_decrement is not None else self.epsilon_min
		self.output_graph=output_graph

		#learning steps counter
		self.learning_step_counter=0

		#initialize memory [s,a,r,s',done] ?añadir done
		self.memory=np.zeros((self.memory_size,n_features*2+2))

		#FF architecture with [target_net, learn_net]
		self._build_model()

		#Retrieving the parameters needed to update the target net
		t_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target_net')
		l_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='learn_net')

		with tf.variable_scope('soft_replacement'):
			self.target_replace_op = [tf.assign(t, l) for t, l in zip(t_params, l_params)]

		self.sess = tf.Session()

		if self.output_graph:
			self.writer=tf.summary.FileWriter('./BOARD/1', self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

	#----------------------------------Inputs----------------------------------
	def create_placeholders(self):
		states=tf.placeholder(tf.float32,[None,self.n_features], name='s')
		next_states=tf.placeholder(tf.float32,[None,self.n_features],name='s_')
		actions=tf.placeholder(tf.int32, [None, ], name='a')
		rewards= tf.placeholder(tf.float32, [None, ], name='r')
		# done=tf.placehloder(tf.bool)
		return states,next_states,actions,rewards


	def _build_model(self):
		self.s,self.s_,self.a,self.r=self.create_placeholders()

		#----------------------------------building learning_net----------------------------------
		with tf.variable_scope('learn_net'):
			l1=tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),
				bias_initializer=tf.constant_initializer(0.1),name='l1')
			self.q_learn=tf.layers.dense(l1,self.n_actions,tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),
				bias_initializer=tf.constant_initializer(0.1),name='q_l')
		#----------------------------------building target_net----------------------------------
		with tf.variable_scope('target_net'):
			t1=tf.layers.dense(self.s_,20,tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),
				bias_initializer=tf.constant_initializer(0.1),name='t1')
			self.q_next=tf.layers.dense(t1,self.n_actions,tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),
				bias_initializer=tf.constant_initializer(0.1),name='t2')

		#Creating the G_T our learner network will try to reach
		with tf.variable_scope('q_target'):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_Target')    # shape=(None, )
			#añadir done b = tf.cond(tf.equal(a, tf.constant(True)), lambda: tf.constant(10), lambda: tf.constant(0))
			self.q_target=tf.stop_gradient(q_target)
		#Now we are going to create the estimated output 
		with tf.variable_scope('q_learn'):
			#one_hot vector with the selected actions for steps training with
			a_indices=tf.one_hot(self.a,depth=self.n_actions,dtype=tf.float32)
			#Storing a tensor with the predicted rewards for those actions by the learning network
			self.q_learn_wrt_a=tf.reduce_sum(self.q_learn*a_indices,axis=1,name='Qmax_learn')

		with tf.variable_scope('loss'):
			self.loss=tf.reduce_mean(tf.squared_difference(self.q_target, self.q_learn_wrt_a, name='loss'))
		with tf.variable_scope('train'):
			self._training_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		#----------------------------------Tracking for TensorBoard----------------------------------
		with tf.variable_scope('learn_net/l1',reuse=True):
			Wl1=tf.get_variable('kernel')
			bl1=tf.get_variable('bias')
		with tf.variable_scope('learn_net/q_l',reuse=True):
			Wl2=tf.get_variable('kernel')
			bl2=tf.get_variable('bias')
		with tf.variable_scope('target_net/t1',reuse=True):
			Wt1=tf.get_variable('kernel')
			bt1=tf.get_variable('bias')
		with tf.variable_scope('target_net/t2',reuse=True):
			Wt2=tf.get_variable('kernel')
			bt2=tf.get_variable('bias')

		tf.summary.histogram("weights",Wl1)
		tf.summary.histogram("weights",Wl2)
		tf.summary.histogram("weights",Wt1)
		tf.summary.histogram("weights",Wt2)

		tf.summary.histogram("biases",bl1)
		tf.summary.histogram("biases",bl2)
		tf.summary.histogram("biases",bt1)
		tf.summary.histogram("biases",bt2)

		tf.summary.histogram("activations",l1)
		tf.summary.histogram("activations",self.q_learn)
		tf.summary.histogram("activations",t1)
		tf.summary.histogram("activations",self.q_next)

		tf.summary.scalar("cost",self.loss)

		self.merged_summary = tf.summary.merge_all()


	def store_transition(self,s,a,r,s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		memory_slice=np.hstack((s,[a,r],s_))
		# replace the old memory with new memory
		# index = self.memory_counter % self.memory_size
		if self.memory_counter >= self.memory_size:
			# make sure we always have some painalities.
			a = self.memory_size* 9 / 10
			b = self.memory_size - a
			if r > 0.0:
				# print(type(a))
				# print(a.size())
				idx = np.random.choice(int(a))
			else:
				# print(type(b))
				# print(b.size())
				idx = np.random.choice(int(b)) + int(a)
		else:
			idx = self.memory_counter
		self.memory[idx, :] = memory_slice
		self.memory_counter+=1

	def choose_action(self,observation,env):
		# to have batch dimension when feed into tf placeholder
		observation = observation[np.newaxis, :]
		if np.random.rand() > self.epsilon:
			actions_value=self.sess.run(self.q_learn,feed_dict={self.s:observation})
			action=np.argmax(actions_value)
		else:
			action=env.action_space.sample() #random action
		return action

	def learn(self):
		if self.learning_step_counter%self.rep_targ_iter == 0:
			self.sess.run(self.target_replace_op)
			# print('\ntarget_params_replaced\n')
		if self.memory_counter>self.memory_size:
			# sample batch memory from all memory
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]
		if self.output_graph:
			summ,_,cost=self.sess.run([self.merged_summary,self._training_op,self.loss],feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features+1],
				self.s_: batch_memory[:, -self.n_features:]
				})
			self.writer.add_summary(summ)
		else:
			_,cost=self.sess.run([self._training_op,self.loss],feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features+1],
				self.s_: batch_memory[:, -self.n_features:]
				})
		#decreasing epsilon
		# new_ep=self.epsilon*self.exploration_decrement
		# self.epsilon= new_ep if new_ep > self.epsilon_min else self.epsilon_min
		self.epsilon = self.epsilon * self.exploration_decrement if self.epsilon > self.epsilon_min else self.epsilon_min
		self.learning_step_counter += 1

if __name__ == '__main__':
	DQN = DeepQNetwork(3,4, output_graph=True)


			
		
