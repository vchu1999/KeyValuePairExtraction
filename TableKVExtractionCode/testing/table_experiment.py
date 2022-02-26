import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame({'d': [1], 'c': [1]})

ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()

plt.savefig('mytable.png')

plt.clf()


