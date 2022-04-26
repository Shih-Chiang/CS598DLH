import matplotlib.pyplot as plt
from matplotlib.figure import Figure


x = range(1,14)
y = [6.5, 4.2, 1.8, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]

fig = plt.figure(figsize=(12, 6))
plt.plot(x, y, 'ko-')
plt.xticks(x)
plt.xlabel('Component Number')
plt.ylabel("Eigenvalue")
plt.title('Scree Plot')
plt.savefig('figure.svg', format='svg')
plt.show()