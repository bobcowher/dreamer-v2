from agent import Agent

agent = Agent(render_mode="rgb_array", max_buffer_size=500000)

agent.collect_dataset(10, use_policy=False)

print("Completed dataset collection")

# agent.train_encoder(epochs=5000, batch_size=64, sequence_length=16)
#
# print("Completed encoder pre-train")

agent.train(epochs=2400)

# del agent

