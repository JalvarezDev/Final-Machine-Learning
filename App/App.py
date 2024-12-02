import gradio as gr
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Función para la búsqueda
def search_images_gradio(query, top_k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    text_embed = model.get_text_features(**inputs)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    
    # Calcular similitud de coseno
    similarities = cosine_similarity(text_embed.detach().numpy(), image_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_images = [unique_images[idx] for idx in top_indices]
    top_scores = [similarities[idx] for idx in top_indices]
    
    # Preparar resultados para mostrar
    results = []
    for img_name, score in zip(top_images, top_scores):
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path)
        results.append((img, f"Similitud: {score:.4f}"))
    
    return results

# Interfaz de Gradio
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <h1 style="text-align:center;">Buscador Semántico de Imágenes</h1>
            <p style="text-align:center;">Escribe una consulta en texto para buscar imágenes relevantes.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Consulta en Texto", placeholder="Describe una imagen...")
                submit_btn = gr.Button("Buscar")
            with gr.Column():
                output_gallery = gr.Gallery(label="Imágenes Relevantes").style(grid=3, height="auto")
        
        # Conectar el botón al método de búsqueda
        submit_btn.click(search_images_gradio, inputs=[query_input], outputs=output_gallery)
    
    return demo

# Crear y lanzar la interfaz
gradio_interface().launch()
