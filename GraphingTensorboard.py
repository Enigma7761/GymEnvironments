import pandas as pd
from matplotlib import pyplot as plt
import tensorboard as tb

print("TensorBoard version: ", tb.__version__)
experiment_id = 'jfaoyIQPTyidZYJuxK5e3A'
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

fig, axs = plt.subplots(1, 2)
axs[0].plot(df['step'][(df['run'] == 'breakout random') & (df['tag'] == 'Scaled_Rewards')], df['value'][(df['run'] == 'breakout random') & (df['tag'] == 'Scaled_Rewards')])
axs[0].plot(df['step'][(df['run'] == 'breakout exploration') & (df['tag'] == 'Scaled_Rewards')], df['value'][(df['run'] == 'breakout exploration') & (df['tag'] == 'Scaled_Rewards')])
axs[0].legend(['Random', 'Model Based'])
axs[0].set_xlabel('Frames')
axs[0].set_ylabel('Scaled Reward')
axs[0].set_title('Breakout')
axs[0].set_xlim(0, 3.363e6)
axs[1].plot(df['step'][(df['run'] == 'pitfall random') & (df['tag'] == 'Scaled_Rewards')], df['value'][(df['run'] == 'pitfall random') & (df['tag'] == 'Scaled_Rewards')])
axs[1].plot(df['step'][(df['run'] == 'pitfall exploration') & (df['tag'] == 'Scaled_Rewards')], df['value'][(df['run'] == 'pitfall exploration') & (df['tag'] == 'Scaled_Rewards')])
axs[1].legend(['Random', 'Model Based'])
axs[1].set_xlabel('Frames')
axs[1].set_ylabel('Scaled Reward')
axs[1].set_title('Pitfall!')
axs[1].set_xlim(0, 2.261e6)

plt.show()