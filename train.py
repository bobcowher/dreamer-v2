from agent import Agent

agent = Agent(render_mode="rgb_array", max_buffer_size=500000)

agent.train(epochs=1200)

# del agent

