# Búsqueda Semántica de Imágenes
Este proyecto implementa una aplicación de búsqueda semántica de imágenes utilizando el modelo CLIP. Permite a los usuarios buscar imágenes relevantes a partir de descripciones en lenguaje natural, devolviendo las imágenes más similares basadas en embeddings calculados por CLIP.

# Características
Modelo base:
openai/clip-vit-base-patch32 para generación de embeddings de texto e imágenes.
Dataset:
Flickr8k, que contiene imágenes y descripciones asociadas.
Interfaz:
Implementada con Gradio para facilitar la interacción.
Implementación optimizada: 
Gestión de embeddings precomputados para mejorar el rendimiento.
Contenerización: 
Configuración lista para ejecutar con Docker.
# Requisitos Previos
## Hardware
CPU o GPU compatible para ejecutar PyTorch.
~4GB de RAM para manejar el modelo y los embeddings.
## Software
Python 3.9 o superior.
Estructura del Proyecto

image-search-app/
├── app.py                 # Código principal de la aplicación
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Documentación del proyecto
├── models/                # Carpeta para almacenar embeddings
│   ├── image_embeddings.npy
│   └── text_embeddings.npy
├── Data/                  # Carpeta con datos del proyecto
│   └── Images/            # Carpeta con las imágenes del dataset
├── NoteBooks

└── ...

# Instalación y Configuración
Crea un entorno virtual e instala las dependencias desde requirements.txt:
python -m venv venv
source venv/bin/activate  # En Linux/Mac
venv\Scripts\activate     # En Windows
pip install -r requirements.txt
## Descargar Dataset
Descarga el dataset Flickr8k desde este enlace.
Extrae las imágenes en la carpeta Data/Images.
## Configurar Embeddings
Si no tienes los embeddings generados, sigue estos pasos:

Abre el notebook proporcionado (generate_embeddings.ipynb) para calcular los embeddings de imágenes y texto.
Guarda los archivos image_embeddings.npy y text_embeddings.npy en la carpeta models/.

# Ejecución
1. Ejecución Local
Ejecuta el archivo app.py:

python app.py
Accede a la aplicación en tu navegador en:
http://127.0.0.1:7860

# Uso de la Aplicación
Ingresa una descripción en lenguaje natural, por ejemplo:

a dog playing in the park
Haz clic en "Buscar".
La aplicación devolverá las imágenes más relevantes junto con la similitud calculada.
# Tecnologías Utilizadas
## Framework de ML: PyTorch.
## Modelo: CLIP (openai/clip-vit-base-patch32).
## Frontend: Gradio para la interfaz de usuario.
## Gestión de Dependencias: pip.

Problemas Comunes
1. El número de imágenes no coincide con los embeddings
Solución: Asegúrate de que el directorio Data/Images contenga exactamente las imágenes para las que se generaron los embeddings.

2. Error InvalidPathError en Gradio
Solución: Asegúrate de agregar la ruta de las imágenes en el parámetro allowed_paths del método launch() en app.py:

python
Copiar código
gradio_interface().launch(allowed_paths=[images_dir])
3. Rendimiento lento
Solución: Precalcula los embeddings y utiliza un subconjunto más pequeño del dataset para pruebas.

