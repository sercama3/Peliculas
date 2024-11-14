import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from thefuzz import process

# Cargar los datos de las películas y calificaciones
@st.cache_data
def cargar_datos():
    movies = pd.read_csv("movies.csv")  # Asegúrate de que contiene 'movieId', 'title' y 'genres'
    ratings = pd.read_csv("ratings.csv")  # Archivo de calificaciones con 'userId', 'movieId' y 'rating'
    return movies, ratings

movies, ratings = cargar_datos()

# Vectorización de géneros para calcular similitud
tfidf = TfidfVectorizer(token_pattern=r'[a-zA-Z0-9]+')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Lista de títulos para coincidencias difusas
lista_titulos = movies['title'].tolist()

# Función para recomendar películas similares basadas en nombre
def recomendar_peliculas_por_nombre(titulo, cosine_sim=cosine_sim):
    if titulo not in movies['title'].values:
        return ["Película no encontrada en el catálogo."]
    
    # Encuentra el índice de la película en el dataset
    idx = movies[movies['title'] == titulo].index[0]
    
    # Obtiene las películas más similares a partir del índice
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Selecciona los 5 más similares excluyendo la película en sí misma
    movie_indices = [i[0] for i in sim_scores[1:6]]
    return movies['title'].iloc[movie_indices]

# Interfaz en Streamlit
st.title("Recomendador de Películas")
st.write("Ingresa el nombre de una película y obtén recomendaciones de películas similares.")

# Entrada de usuario
titulo_ingresado = st.text_input("Escribe el nombre de la película")

# Inicializar la variable del selectbox
pelicula_seleccionada = None

# Usamos session_state para almacenar el estado
if "pelicula_seleccionada" not in st.session_state:
    st.session_state.pelicula_seleccionada = None

# Botón para buscar películas que coincidan con el texto ingresado
if st.button("Buscar Películas"):
    if titulo_ingresado:
        # Filtrar películas usando coincidencias difusas
        coincidencias = process.extract(titulo_ingresado, lista_titulos, limit=5)
        peliculas_filtradas = [match[0] for match in coincidencias if match[1] >= 60]
        
        # Mostrar el selectbox si hay coincidencias
        if peliculas_filtradas:
            st.session_state.pelicula_seleccionada = st.selectbox("Selecciona una película de la lista:", peliculas_filtradas)
        else:
            st.write("No se encontraron películas que coincidan con tu búsqueda.")
    else:
        st.write("Por favor, ingresa el nombre de una película para buscar.")

# Mostrar el botón para obtener recomendaciones solo si una película ha sido seleccionada
if st.session_state.pelicula_seleccionada:
    if st.button("Obtener Recomendaciones"):
        recomendaciones = recomendar_peliculas_por_nombre(st.session_state.pelicula_seleccionada)
        st.write("Películas recomendadas:")
        for i, rec in enumerate(recomendaciones, 1):
            st.write(f"{i}. {rec}")


