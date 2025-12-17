
SICP - Aplicativo de Escritorio 

Herramienta de auditoría algorítmica para la contratación pública.

 Funcionalidades del Software

El programa cuenta con tres módulos integrados en una interfaz gráfica:

Datos y EDA: Conecta con la API de datos.gov.co, descarga la información y genera gráficos descriptivos sobre modalidades y montos.
Segmentación (K-Means): Ejecuta el motor de Inteligencia Artificial para agrupar entidades en 4 perfiles de comportamiento.
Auditoría y Consulta: Buscador inteligente que permite localizar una entidad (por nombre o NIT) y verificar si pertenece al clúster de "Alto Riesgo".

 Cómo ejecutar la aplicación

Prerrequisitos
Asegúrese de estar en la carpeta raíz del proyecto e instalar las dependencias:

pip install -r ../requirements.txt

Ejecución

Navegue a esta carpeta y ejecute el script principal:

cd app_escritorio
python programa1.1.py


Notas Técnicas

El software crea archivos temporales .pkl para guardar el modelo entrenado.
Requiere conexión a internet activa para la descarga de datos desde la API SODA.
