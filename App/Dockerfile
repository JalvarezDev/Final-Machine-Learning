# Imagen base de Python
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Crear directorio de la app
WORKDIR /app

# Copiar archivos al contenedor
COPY requirements.txt requirements.txt
COPY app.py app.py

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la app
EXPOSE 7860

# Comando para ejecutar la app
CMD ["python", "app.py"]
