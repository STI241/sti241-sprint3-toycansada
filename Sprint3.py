import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Calcular KNN
def calcular_vecinos():
    # Usuario (p) a comparar y cantidad de vecinos (k)
    p = combobox_usuarios.get()
    k = int(combobox_vecinos.get())

    # NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(dataframe_scaled)

    p_index = dataframe.index.get_loc(p)
    distances, indices = nn.kneighbors([dataframe_scaled[p_index]])

    # Limpiar el área de texto antes de mostrar los nuevos resultados
    text_area.delete(1.0, tk.END)
    
    # KNN y distancias en consola
    text_area.insert(tk.END, f"Vecinos más cercanos para {p}:\n")
    for j, idx in enumerate(indices[0]):
        if j == 0:
            continue
        vecino = dataframe.index[idx]
        distancia = distances[0][j]
        text_area.insert(tk.END, f"{vecino} - Distancia: {distancia:.2f}\n")

    # KNN en gráfico
    vecinos = [dataframe.index[idx] for idx in indices[0]]
    distancias_vecinos = distances[0]

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(dataframe_scaled)
    df_pca = pd.DataFrame(data_pca, index=dataframe.index, columns=['PC1', 'PC2'])

    # Grafica
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], label='Usuarios')
    plt.scatter(df_pca.loc[vecinos, 'PC1'], df_pca.loc[vecinos, 'PC2'], color='red', label='Vecinos más cercanos')
    plt.scatter(df_pca.loc[p, 'PC1'], df_pca.loc[p, 'PC2'], color='green', label=f'{p} (Usuario de interés)')
    plt.title(f'Vecinos más cercanos a {p}')
    for i, vecino in enumerate(vecinos):
        plt.annotate(f'{vecino}\nDistancia: {distancias_vecinos[i]:.2f}', (df_pca.loc[vecino, 'PC1'], df_pca.loc[vecino, 'PC2']), xytext=(5,-5), textcoords='offset points')
    plt.legend()
    plt.show()

# Cargar datos
data = pd.read_csv('data.csv', index_col=0)
dataframe = data.transpose()

# Normalizar características
scaler = StandardScaler()
dataframe_scaled = scaler.fit_transform(dataframe)

# Interfaz
root = tk.Tk()
root.title("Buscar Vecinos Más Cercanos")

main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=10)

ttk.Label(main_frame, text="Usuario de interés (p):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
usuarios = dataframe.index.tolist()
combobox_usuarios = ttk.Combobox(main_frame, values=usuarios, state="readonly")
combobox_usuarios.grid(row=0, column=1, padx=5, pady=5)

# Ccantidad de vecinos posibles
cantidad_vecinos = len(usuarios) - 1
vecinos = list(range(1, cantidad_vecinos + 1))

# Seleccionar la cantidad de vecinos
ttk.Label(main_frame, text="Cantidad de vecinos (k):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
combobox_vecinos = ttk.Combobox(main_frame, values=vecinos, state="readonly")
combobox_vecinos.grid(row=1, column=1, padx=5, pady=5)

# Botón para calcular
calcular_button = ttk.Button(main_frame, text="Calcular Vecinos", command=calcular_vecinos)
calcular_button.grid(row=2, column=0, columnspan=2, pady=10)

# Área de texto para mostrar los vecinos más cercanos
text_area = scrolledtext.ScrolledText(main_frame, width=40, height=10, wrap=tk.WORD)
text_area.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()