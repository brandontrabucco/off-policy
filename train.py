from offpolicy import train
import gym


if __name__ == "__main__":

    train('./half_cheetah_td3',
          gym.make('HalfCheetah-v2'),
          gym.make('HalfCheetah-v2'),
          'TD3')
