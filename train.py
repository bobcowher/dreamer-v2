from agent import Agent

agent = Agent(render_mode="rgb_array", max_buffer_size=50000)

agent.train(epochs=1000)

# del agent

