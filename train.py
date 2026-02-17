from agent import Agent

agent = Agent(render_mode="rgb_array", max_buffer_size=500000)

agent.collect_dataset(10, use_policy=False)

print("Completed dataset collection")

# agent.train_encoder(epochs=5000, batch_size=64, sequence_length=16)
#
# print("Completed encoder pre-train")

world_model_epochs = 10
actor_critic_epochs = 10

agent.train(epochs=2400, 
            wm_epochs=world_model_epochs, 
            ac_epochs=actor_critic_epochs, 
            summary_writer_label=f"wm_ep={world_model_epochs}-ac_ep={actor_critic_epochs}")

# del agent

