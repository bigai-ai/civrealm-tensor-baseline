class Runner:
    def __init__(self):
        super().__init__()

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
