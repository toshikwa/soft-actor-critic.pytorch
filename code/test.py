def test(self):
    # restore
    self.agent.load_model(os.path.join(self.logdir, "model"))
    state = self.env.reset()
    done = False
    while not done:
        self.env.render()
        action = self.agent.select_action(state, eval=True)
        next_state, _, done, _ = self.env.step(action)
        state = next_state