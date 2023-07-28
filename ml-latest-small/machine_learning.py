import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Carga el dataset de ratings de MovieLens
ratings_file = "ratings.csv"
movies_file = "movies.csv"

ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

# Combinar el dataset de películas y calificaciones por el ID de película
movies_ratings = pd.merge(ratings, movies, on='movieId')

# Crear una tabla pivote de usuarios y películas
user_movie_ratings = movies_ratings.pivot_table(index='userId', columns='title', values='rating')

# Rellenar los valores faltantes con 0
user_movie_ratings = user_movie_ratings.fillna(0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X = user_movie_ratings.values.T
y = user_movie_ratings.columns
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo k-NN
model_knn = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
model_knn.fit(X_train)


def get_movie_recommendations(liked_movies, n_recommendations=5):
    user_ratings = pd.Series(liked_movies)
    movie_indices = []

    for movie in user_ratings.index:
        movie_info = movies[movies['title'].str.contains(str(movie), case=False, na=False)]
        if not movie_info.empty:
            movie_index = movie_info.index[0]
            movie_indices.append(movie_index)
        else:
            messagebox.showinfo("Advertencia", f"La película '{movie}' no se encuentra en el dataset y será ignorada.")
            print(f"La película '{movie}' no se encuentra en el dataset y será ignorada.")

    if not movie_indices:
        messagebox.showinfo("Error", "No se encontraron películas válidas en el dataset.")
        print("No se encontraron películas válidas en el dataset.")
        return []

    print(f"Películas ingresadas: {liked_movies}")
    print(f"Índices de películas encontradas: {movie_indices}")

    # Utilizar los índices de películas encontradas para obtener las características relevantes
    query = X_train[movie_indices, :]

    # Promediar las características de las películas ingresadas por el usuario
    query_combined = np.mean(query, axis=0, keepdims=True)

    # Asegurarnos de que n_recommendations sea un valor numérico válido
    n_neighbors = min(n_recommendations, len(X_train) - 1)

    distances, indices = model_knn.kneighbors(query_combined, n_neighbors=n_neighbors)

    recommended_movies = []
    for i in range(n_neighbors):
        movie_index = indices[0, i]
        movie_title = user_movie_ratings.columns[movie_index]
        recommended_movies.append(movie_title)

    return recommended_movies


# Función para mostrar las recomendaciones en una ventana
def show_recommendations_window():
    user_input_movies = [movie.strip() for movie in user_input_entry.get().split(',')]  # Eliminar espacios en blanco
    try:
        n_recommendations = int(n_recommendations_entry.get())  # Convertir el valor a entero
        recommendations = get_movie_recommendations(user_input_movies, n_recommendations)

        # Crear una ventana para mostrar las recomendaciones
        recommendations_window = tk.Toplevel(root)
        recommendations_window.title("Recomendaciones")
        recommendations_window.geometry("400x300")

        recommendations_label = tk.Label(recommendations_window, text="¡Recomendaciones para ti!", font=("Helvetica", 16, "bold"))
        recommendations_label.pack(pady=10)

        for i, movie in enumerate(recommendations):
            movie_label = tk.Label(recommendations_window, text=f"{i+1}. {movie}", font=("Helvetica", 12))
            movie_label.pack()

        # Reducción de dimensionalidad utilizando t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_train)

        # Obtener las coordenadas del usuario de consulta
        user_input_indices = [y.get_loc(movie) for movie in user_input_movies if movie in y and y.get_loc(movie) < X_tsne.shape[0]]
        user_input_coords = X_tsne[user_input_indices]

        # Obtener las coordenadas de las películas recomendadas
        user_input_ratings = [5] * len(user_input_movies)  # Valor de las calificaciones que el usuario ingresa
        query_combined = np.mean(X_train[user_input_indices], axis=0, keepdims=True)
        _, recommendation_indices = model_knn.kneighbors(query_combined, n_neighbors=n_recommendations)
        recommendation_indices = [index for index in recommendation_indices[0] if index < X_tsne.shape[0]]
        recommendation_coords = X_tsne[recommendation_indices]

        # Visualización
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.2, label='Películas')
        plt.scatter(user_input_coords[:, 0], user_input_coords[:, 1], color='red', marker='*', s=150, label='Películas seleccionadas por el usuario')
        plt.scatter(recommendation_coords[:, 0], recommendation_coords[:, 1], color='green', marker='o', s=100, label='Películas recomendadas')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.title('Visualización t-SNE de Películas')
        plt.legend()
        plt.show()

    except ValueError:
        messagebox.showinfo("Error", "Por favor, ingresa un número válido de recomendaciones.")


# Crear la ventana principal
root = tk.Tk()
root.title("Chat de Recomendación de Películas")
root.geometry("400x250")

# Etiqueta e input para ingresar las películas que le gustan al usuario
input_label = tk.Label(root, text="Por favor, ingresa el nombre de algunas películas que te gusten (separadas por comas):")
input_label.pack(pady=10)

user_input_entry = tk.Entry(root, width=40)
user_input_entry.pack(pady=5)

# Input para ingresar el número de recomendaciones deseado
n_recommendations_label = tk.Label(root, text="Número de recomendaciones:")
n_recommendations_label.pack(pady=5)

n_recommendations_entry = tk.Entry(root, width=10)
n_recommendations_entry.pack(pady=5)
n_recommendations_entry.insert(0, "5")  # Valor predeterminado, puedes cambiarlo según prefieras

# Botón para obtener las recomendaciones
recommend_button = tk.Button(root, text="Obtener Recomendaciones", command=show_recommendations_window)
recommend_button.pack(pady=10)

root.mainloop()
