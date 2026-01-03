from agent import Agent

agent = Agent(render_mode="human", max_buffer_size=50000)

agent.load_models()

agent.evaluate_policy()

# del agent

