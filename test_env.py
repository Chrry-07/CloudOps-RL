from env import CloudOpsEnv

env = CloudOpsEnv()

state = env.reset()
print("Initial:", state)

next_state, reward, done = env.step("scale_up")
print("After action:", next_state)
print("Reward:", reward)
print("Done:", done)