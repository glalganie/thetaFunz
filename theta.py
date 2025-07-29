import numpy as np
import matplotlib.pyplot as plt

# Funzione di Costo Y(θ) = (θ - 3)^2
def cost_function(theta):
    return (theta - 3)**2

# Derivata della funzione di costo Y'(θ) = 2(θ - 3)
def gradient(theta):
    return 2 * (theta - 3)

# Parametri Gradient Descent
learning_rate = 0.1  # Tasso di apprendimento
iterations = 100  # Numero di iterazioni
theta = 0  # Inizializzazione di θ

# Memorizziamo i valori di θ e Y(θ) per il grafico
theta_values = [theta]
cost_values = [cost_function(theta)]

# Gradient Descent
for i in range(iterations):
    # Calcolo del gradiente
    grad = gradient(theta)
    
    # Aggiornamento di θ
    theta = theta - learning_rate * grad
    
    # Memorizziamo i valori per il grafico
    theta_values.append(theta)
    cost_values.append(cost_function(theta))

# Grafico della funzione di costo
theta_range = np.linspace(-1, 7, 400)
cost_range = cost_function(theta_range)


plt.legend()
plt.savefig('gradient_descent_plot.png') # Salva il grafico come file PNG

# plt.show() # Rimuovi o commenta questa riga per evitare di aprire una finestra separata


plt.plot(theta_range, cost_range, label='Funzione di Costo', color='b')
plt.plot(theta_values, cost_values, label='Gradient Descent', color='r', marker='o')
plt.title('Gradient Descent per Minimizzare la Funzione di Costo')
plt.xlabel('θ')
plt.ylabel('Y(θ)')
plt.legend()
plt.show()

# Risultato finale
print(f"Valore finale di θ: {theta}")
print(f"Funzione di Costo finale: {cost_function(theta)}")
