import gymnasium as gym
import browsergym.workarena  # register workarena tasks as gym environments
import browsergym

env = gym.make("browsergym/workarena.servicenow.order-ipad-pro")
obs, info = env.reset()
done = False
# env.action_spaces is a Unicode (subclass of Text) space with max_length = 2**32 - 1,
# which is very funny and causes a timeout when calling sample()
space = browsergym.core.spaces.Unicode(50, min_length=0)
for _ in range(5):
    action = "click('1')"
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    done = terminated or truncated