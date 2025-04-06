from diffusion.diffusion_base import linear_beta_schedule, cosine_beta_schedule
import matplotlib.pyplot as plt
import numpy as np
import torch

timesteps = 15
# print(np.arange(0, 16))

plt.plot(1- linear_beta_schedule(timesteps), label = "Linear")
plt.plot(1- cosine_beta_schedule(timesteps), label = "Cosine")
plt.xlabel("diffusion steps")
plt.ylabel(r"$\beta_t$")
plt.legend()
plt.show() 

plt.plot(linear_beta_schedule(timesteps), label = "Linear")
plt.plot(cosine_beta_schedule(timesteps), label = "Cosine")
plt.xlabel("diffusion steps")
plt.ylabel(r"$\alpha_t$")
plt.legend()
plt.show()

plt.plot(np.cumprod(linear_beta_schedule(timesteps)), label = "Linear")
plt.plot(np.cumprod(cosine_beta_schedule(timesteps)), label = "Cosine")
plt.xlabel("diffusion steps")
plt.ylabel(r"$\bar \alpha_t$")
plt.legend()
plt.show()

