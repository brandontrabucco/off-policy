from offpolicy import train
import gym


if __name__ == "__main__":

    train('./ant_td3',
          gym.make('Ant-v2'),
          gym.make('Ant-v2'),
          'TD3')
