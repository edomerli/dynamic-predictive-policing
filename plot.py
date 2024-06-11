import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("df_fair")

fig = plt.figure()
print(df.head())

eq_discovery = df["equality_discovery_prob"]
for i in range(len(eq_discovery)):
    print(eq_discovery[i])
    eq_discovery[i] = eq_discovery[i][0]
print(eq_discovery)
# plt.plot(df["prevented_crimes"], label="Prevented crimes")
# plt.plot(df["committed_crimes"], label="Committed crimes")
plt.plot(df["equality_discovery_prob"], label="Equality of discovery probability")
plt.legend()

plt.show()