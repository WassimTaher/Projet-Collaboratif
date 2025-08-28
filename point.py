import matplotlib.pyplot as plt
import numpy as np

#Ce code ne sert qu'a savoir quel fichier comporte quel partie du maillage
# Liste des couleurs à utiliser pour chaque fichier
colors = plt.cm.tab20(np.linspace(0, 1, 7))  # 21 couleurs distinctes

# Créer une figure
plt.figure(figsize=(12, 8))

# Parcourir les fichiers de point.txt à point7.txt
for i in range(1, 7):
    filename = f'point{i}.txt'
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Ignorer la première ligne (nombre de points)
        lines = lines[1:]

        # Extraire les points x et y
        x = []
        y = []
        for line in lines:
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
        
        # Tracer les points avec une couleur différente
        plt.plot(x, y, marker='o', markersize=2, linestyle='-', color=colors[i-1], label=f'Fichier {i}')
    
    except FileNotFoundError:
        print(f"Le fichier {filename} n'a pas été trouvé.")

# Ajouter des labels et une légende
plt.title('Tracé des points de points1.txt à points6.txt')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Légende à droite
plt.tight_layout()  # Ajuster la mise en page
plt.show()
