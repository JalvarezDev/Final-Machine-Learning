import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

# Cargar embeddings
try:
    image_embeddings = np.load("models/image_embeddings.npy")
    print(f"Embeddings de imágenes cargados. Dimensiones: {image_embeddings.shape}")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'image_embeddings.npy'. Asegúrate de que esté en la carpeta 'models'.")
    exit()

# Obtener la ruta del directorio del proyecto (subir un nivel)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construir la ruta a "Data/Images"
images_dir = os.path.join(project_dir, "Data", "Images")

# Listar las imágenes en el directorio
try:
    unique_images = os.listdir(images_dir)
    print(f"Imágenes disponibles: {len(unique_images)}")
except FileNotFoundError:
    print(f"Error: No se encontró el directorio de imágenes en '{images_dir}'. Asegúrate de que la carpeta exista.")
    exit()

# Verificar que el número de imágenes coincida con los embeddings
if len(unique_images) > len(image_embeddings):
    unique_images = unique_images[:len(image_embeddings)]
    print(f"Se han reducido las imágenes a {len(unique_images)} para coincidir con los embeddings.")

# Cargar modelo y procesador CLIP
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Modelo CLIP cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo CLIP: {e}")
    exit()

# Función para generar embeddings del texto
def generate_text_embedding(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    text_embed = model.get_text_features(**inputs)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    return text_embed.detach().numpy()

# Función de búsqueda
def search_images(query, top_k=5, threshold=0.1):
    print(f"Búsqueda para la consulta: '{query}'")
    text_embed = generate_text_embedding(query)
    similarities = cosine_similarity(text_embed, image_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        if i < len(unique_images) and similarities[i] >= threshold:  # Verificar índice y umbral
            img_path = os.path.join(images_dir, unique_images[i])
            results.append((img_path, f"Similitud: {similarities[i]:.4f}"))
    
    if not results:
        print("No se encontraron resultados relevantes.")
    return results

# Interfaz Gradio
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Búsqueda Semántica de Imágenes")
        query_input = gr.Textbox(label="Describe una imagen...")
        output_gallery = gr.Gallery(label="Imágenes Relevantes")
        submit_btn = gr.Button("Buscar")

        submit_btn.click(search_images, inputs=[query_input], outputs=output_gallery)
        
    return demo

if __name__ == "__main__":
    gradio_interface().launch(allowed_paths=[images_dir])

