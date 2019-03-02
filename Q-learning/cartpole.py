import numpy as np
import gym
from keras.models import Model
import keras.layers as layers
from keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


class QAgent:
    """Q learning agent.
    Holds the model for the Q-function and the memory for memory replay.
    Contains a function to fit the model from memory.
    Contains a function to save state-action-reward-state tuples to the memory.
    Contains a function to choose an action given a state
    """
    def __init__(self, memory_size=5000, discount=.98):

        self.action_dim = 2  # action 0: push left; action 1: push right
        self.state_dim = 4

        self.discount = discount

        self.avg_mse = 0  # average mean square error for value prediction. needed for selective memory
        self.avg_pred_variance = 0.001

        self.Q_model_train, self.Q_model_predict = self.build_q_model()
        self.memory = deque(maxlen=memory_size)

    def get_action(self, st, rand=0):
        qs = self.Q_model_predict.predict(st.reshape(1, self.state_dim))
        noise = np.random.normal(loc=0, scale=rand, size=self.action_dim)
        return np.argmax(qs + noise)

    def remember_episode(self, st1, ac, rew, st2, don):
        """Commit series to memory.
        Selective memory: only memorize samples where prediction was worse than average.
        This means the actual Q-values must be known.
        """
        actual_qs = self.calc_q(rew)
        predicted_qs = self.Q_model_predict.predict(st1) * np.array(ac)
        predicted_qs = np.sum(predicted_qs, axis=1)
        mse = (predicted_qs - actual_qs) ** 2
        pick = mse > 1.1 * self.avg_mse  # pick samples with high mean square error
        self.avg_mse = 0.99 * self.avg_mse + 0.01 * np.mean(mse)

        mem_list = [st1[pick].tolist(),
                    ac[pick].tolist(),
                    rew[pick].tolist(),
                    st2[pick].tolist(),
                    don[pick].tolist()
                    ]
        transposed_list = map(list, zip(*mem_list))
        self.memory.extend(transposed_list)

        return np.sum(pick)

    def calc_q(self, rew):

        """Calculate the actual Q-values from a series of rewards.
        """
        ll = len(rew)
        disc = self.discount ** np.arange(ll)
        qs = np.outer(1 / disc, rew * disc)
        qs = np.triu(qs)  # upper triangle
        qs = np.sum(qs, axis=1)
        return qs

    def fit_from_memory(self, epochs=3, batch_size=16):
        """Fit the Q model on random data from the memory
        """
        sample_size = min(batch_size, len(self.memory))
        for i in range(epochs):
            sample = random.sample(self.memory, sample_size)
            st1, ac, rew, st2, don = map(np.array, zip(*sample))
            q_next = self.Q_model_predict.predict(st2)
            q_next = np.max(q_next, axis=1) * (1 - don)
            q_target = rew + self.discount * q_next
            q_target = np.vstack((q_target, q_target)).T
            q_target = q_target * ac
            self.Q_model_train.train_on_batch([st1, ac], q_target)

    def build_q_model(self, hidden_layer_sizes=(40, 40)):
        """build the Q model.
        Returns one model for prediction and one for training.
        """
        inp_st = layers.Input(shape=(self.state_dim,))
        prev = inp_st
        for n in hidden_layer_sizes:
            prev = layers.Dense(n, activation='relu')(prev)
        out_ac = layers.Dense(self.action_dim)(prev)

        """we only want to fit the output for the action actually taken.
        We contract the predicted outputs with a mask that is to be provided by input when training.
        """
        inp_mask = layers.Input(shape=(self.action_dim,))
        out_masked = layers.Multiply()([out_ac, inp_mask])

        model_train = Model(inputs=[inp_st, inp_mask], outputs=out_masked)
        model_train.compile(loss='mse', optimizer=Adam())

        model_predict = Model(inputs=inp_st, outputs=out_ac)

        return model_train, model_predict


class Session:
    """holds the agent
    provides simple to use functions for training, testing and plotting
    """
    def __init__(self):
        self.agent = QAgent()

        self.achievements = []
        self.new_mems = []
        self.variances = []

        self.total_train_time = 0

    def training_cycle(self, epochs=5, epoch_time=10000, plotting=False):
        """execute multiple epochs of training.
        between the epochs, a progress bar is updated and, if plotting=True, plots are shown
        """
        for _ in tqdm(range(epochs)):
            self.train(approx_train_time=epoch_time)
            if plotting:
                self.plot_data()
        print('Total train time: {}'.format(self.total_train_time))
        print('Number of completed episodes: {}'.format(len(self.achievements)))

    def plot_data(self):
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.achievements, 'r.')
        plt.title('achievements')
        plt.subplot(1, 3, 2)
        plt.plot(self.variances, 'r-')
        plt.title('average prediction error')
        plt.subplot(1, 3, 3)
        plt.plot(self.new_mems, 'r.')
        plt.title('number of new memories')
        plt.show()

    def play(self, n_rounds=1, render=True):
        """play 'n_rounds' rounds of CartPole-v1.
        use 'agent' to choose an action at each timestep.
        if 'render' is True, render the game to a separate window.
        return a list of the total reward achieved in each round.
        """
        achievements = []
        env = gym.make('CartPole-v1')

        for _ in range(n_rounds):

            observation = env.reset()
            if render:
                env.render()

            cum_reward = 0
            done = False

            while not done:
                action = self.agent.get_action(observation, rand=0)
                observation, reward, done, _ = env.step(action)
                cum_reward += reward

                if render:
                    env.render()

            achievements.append(cum_reward)

        env.close()
        return achievements

    def train(self, approx_train_time=10000):
        """train 'agent' for 'approx_train_time' time steps of CartPole-v1.
        the time is only approximate because training will only end after completing an episode,
        this is needed for selective memory.
        """
        action_list = np.identity(self.agent.action_dim)

        env = gym.make('CartPole-v1')

        achievements = []
        variances = []
        new_mems = []
        cum_reward = 0
        observation = env.reset()
        st1 = []
        ac = []
        rew = []
        st2 = []
        don = []

        t = 0
        running = True
        while running:
            t += 1

            st1.append(observation)

            rand = min(2, np.sqrt(self.agent.avg_mse) / 2)
            action = self.agent.get_action(observation, rand)
            observation, reward, done, _ = env.step(action)
            cum_reward += reward

            ac.append(action_list[action])
            rew.append(reward)
            st2.append(observation)
            don.append(done)

            if done:
                if t > approx_train_time:
                    running = False
                achievements.append(cum_reward)

                # only learn from failed episodes because we dont know the future reward of the last state
                if cum_reward < 500:
                    n_mems = self.agent.remember_episode(np.array(st1), np.array(ac),
                                                         np.array(rew), np.array(st2), np.array(don))
                    self.agent.fit_from_memory(epochs=4, batch_size=64)
                else:
                    n_mems = 0

                variances.append(np.sqrt(self.agent.avg_mse))
                new_mems.append(n_mems)

                st1 = []
                ac = []
                rew = []
                st2 = []
                don = []
                cum_reward = 0
                observation = env.reset()

        env.close()
        self.achievements += achievements
        self.variances += variances
        self.new_mems += new_mems
        self.total_train_time += t
        self.variances.append(np.sqrt(self.agent.avg_mse))


if __name__ == "__main__":
    """Minimal example
    trains the agent, and afterwards renders how the agent plays one round
    lastly, data collected during training is plotted
    """
    epochs = 10
    epoch_time = 10000
    print("training the agent for {} epochs of approximately {} time steps each...".format(epochs, epoch_time))
    sess = Session()
    sess.training_cycle(epochs, epoch_time, plotting=False)
    sess.play()
    sess.plot_data()
