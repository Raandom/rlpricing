
class RandomModel(object):
    def __init__(self, _, env, *args, **kwargs):
        self.env = env

    def predict(self, *args, **kwargs):
        return [self.env.action_space.sample()], None

    def learn(self, total_timesteps, *args, **kwargs):
        self.env.reset()
        for i in range(0, total_timesteps):
            if i % 10000 == 0:
                print("Running step {}".format(i))
            self.env.step([self.env.action_space.sample()])

    def get_periodic_profits(self):
        return 0
