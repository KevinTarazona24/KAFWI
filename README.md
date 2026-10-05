# KAFWI

## Descripción del repositorio

Este repositorio contiene códigos en Python para estimar modelos bidimensionales de velocidad de la onda P a partir de datos de sísmica de reflexión, mediante aprendizaje profundo no supervisado guiado por la física.

El aporte principal de la investigación consiste en el diseño de una arquitectura neuronal de dos ramas que combina información complementaria de los registros sísmicos para estimar el modelo de velocidad. Este modelo se integra con un operador acústico diferenciable, que permite simular su respuesta sísmica y calcular los gradientes necesarios para optimizar los parámetros de la red.

El proyecto forma parte de una tesis de pregrado en Geología de la **Universidad Industrial de Santander (UIS)**.

## Tesis y diapositivas de la presentación

El documento de tesis y las diapositivas de sustentación se incorporarán a este apartado cuando estén disponibles.

## Divulgación académica

Durante el desarrollo de esta investigación se generaron productos de divulgación académica para compartir la metodología, los avances y los principales resultados.

### Participación en STSIVA 2026

La investigación fue presentada en el **XXVI International Symposium of Image, Signal Processing, and Artificial Vision (STSIVA 2026)**, realizado del **2 al 4 de septiembre de 2026**, con el artículo de conferencia titulado:

**“Unsupervised Physics-Constrained Neural Approximation for Full-Waveform Inversion”.**

### Presentación en la Sexta Semana de la Geofísica

Los avances y principales resultados de la investigación serán presentados durante la **Sexta Semana de la Geofísica de la Universidad Industrial de Santander**, que se llevará a cabo del **3 al 7 de noviembre de 2026**, con la presentación titulada:

**“Estimación de modelos de velocidad de la onda P en cuencas emergentes colombianas y sus márgenes estructurales”.**

## Base de datos

La investigación utiliza datos sísmicos sintéticos y sus correspondientes modelos de velocidad de referencia para evaluar la recuperación de estructuras del subsuelo.

Los escenarios considerados incluyen las familias **FlatVel, CurveVel, FaultVel y Style de OpenFWI**, así como escenarios geológicos de **GeoFWI**.

La información utilizada comprende:

- **Registros sísmicos:** respuestas de varios disparos registradas en un conjunto de receptores.
- **Modelos de velocidad de referencia:** distribuciones bidimensionales de velocidad de la onda P utilizadas para evaluar las estimaciones.
- **Parámetros de adquisición y simulación:** posiciones de fuentes y receptores, espaciamiento de la malla, intervalo temporal y características de la ondícula fuente.

En el esquema no supervisado, los modelos de referencia se utilizan para evaluar los resultados; no se emplean como etiquetas objetivo en la función de pérdida.

Los conjuntos de datos deben obtenerse por separado y sus rutas deben configurarse antes de ejecutar los experimentos.

## Organización del repositorio

```text
KAFWI/
├── AcousticOperator/
├── Unsupervised_Approach/
├── main.py
├── utils.py
├── rainbow256.npy
├── README.md
└── .gitignore
```

- **AcousticOperator/**: componentes relacionados con el operador acústico y la simulación de la propagación de ondas.
- **Unsupervised_Approach/**: componentes del enfoque de estimación neuronal y optimización no supervisada.
- **main.py**: script principal para la configuración y ejecución de los experimentos.
- **utils.py**: funciones auxiliares de la implementación.
- **rainbow256.npy**: recurso auxiliar en formato NumPy.
- **README.md**: descripción y documentación general del proyecto.
- **.gitignore**: reglas para excluir archivos temporales y otros recursos del seguimiento de Git.

## Requisitos previos

Para ejecutar los códigos se requiere un entorno de Python con las dependencias utilizadas por los scripts.

Antes de iniciar un experimento, se debe:

1. Instalar las bibliotecas importadas por los códigos.
2. Obtener los registros sísmicos y los modelos de velocidad correspondientes.
3. Configurar las rutas de entrada.
4. Revisar los parámetros de adquisición y simulación.
5. Configurar el dispositivo de cómputo y los parámetros de optimización.

Los recursos computacionales necesarios dependen del tamaño del modelo, el número de disparos, la duración de los registros y la configuración del operador acústico.

## Resumen de la metodología

### 1. Preparación de los datos sísmicos

Los registros sísmicos se organizan según la geometría de adquisición y se preparan como entrada de la arquitectura neuronal.

Para comparar los datos observados y simulados, se deben mantener consistentes las posiciones de las fuentes y los receptores, el muestreo espacial y temporal, y la definición de la fuente sísmica.

### 2. Estimación mediante una arquitectura de dos ramas

La arquitectura procesa información complementaria de los registros sísmicos a través de dos ramas y combina las características extraídas para estimar un modelo bidimensional de velocidad de la onda P.

El diseño de esta arquitectura constituye el aporte principal del trabajo dentro del esquema de aprendizaje profundo no supervisado guiado por la física.

### 3. Simulación con un operador acústico diferenciable

El modelo de velocidad estimado se introduce en el operador acústico para generar registros sísmicos simulados.

La implementación diferenciable permite calcular gradientes a través de la simulación, relacionando la discrepancia entre los registros con el modelo de velocidad estimado y los parámetros de la red neuronal.

### 4. Optimización no supervisada

La función de pérdida combina:

- **Desajuste L1:** mide las diferencias absolutas entre los registros observados y simulados.
- **Desajuste L2:** penaliza las diferencias cuadráticas entre ambos registros.
- **Variación total anisotrópica:** regulariza las variaciones espaciales del modelo de velocidad.

Además, se incorpora un precondicionamiento del gradiente dependiente de la profundidad para ajustar su contribución durante la optimización.

### 5. Evaluación de los resultados

La evaluación considera tanto la correspondencia entre los modelos estimados y los modelos de referencia como el ajuste entre los registros sísmicos observados y simulados.

Desde la interpretación geológica, se analiza la recuperación de la geometría de las capas, los contrastes de velocidad y las estructuras plegadas, incluidas antiformas y sinformas.

## Cómo obtener el repositorio

Para descargar una copia local, ejecutar en una terminal:

```bash
git clone https://github.com/KevinTarazona24/KAFWI.git
cd KAFWI
```

Antes de ejecutar los códigos, revisar la configuración del experimento y ajustar las rutas de los datos y los parámetros correspondientes.

## ¿Cómo colaborar con el proyecto?

Puedes contribuir reportando errores, proponiendo mejoras en la documentación o compartiendo observaciones que faciliten la reproducción de los experimentos.

Para reportar un problema, abre un **issue** en GitHub e incluye:

- Una descripción del problema.
- El script y la configuración utilizados.
- El mensaje de error, cuando corresponda.
- Los pasos necesarios para reproducirlo.

Las propuestas de modificación del código pueden enviarse mediante un **pull request**.

## Créditos

- **Kevin Tarazona:** estudiante de Geología de la Universidad Industrial de Santander y miembro activo del Semillero de Investigación en Geofísica Aplicada y Computacional.
- **Dirección de tesis:** Ms.c Ana Gabriela Mantilla Dulcey y Ph(D) Yesid Paul Goyes Peñafiel. 
- **Colaboradores de la investigación:** Estudiante de Doctorado Javier Torres Quintero.

Los conjuntos de datos, las bibliotecas y los métodos externos utilizados deben reconocerse mediante sus respectivas publicaciones y repositorios originales.

## Cómo citar

Si utilizas los códigos o resultados de este proyecto en un trabajo académico, cita la tesis asociada:

> Tarazona, K. ([2026]). *[Inversión de onda completa mediante aprendizaje profundo no supervisado guiado por la física]* [Tesis de pregrado, Universidad Industrial de Santander].

Si utilizas contenidos relacionados con el artículo presentado en STSIVA 2026, incluye también su referencia:

> [Lista completa de autores]. (2026). *Unsupervised Physics-Constrained Neural Approximation for Full-Waveform Inversion*. XXVI International Symposium of Image, Signal Processing, and Artificial Vision (STSIVA). [DOI o enlace de publicación].

## Licencia

La licencia de uso de este proyecto está pendiente de definición y se incorporará en el archivo `LICENSE`.

Los datos y las bibliotecas de terceros conservan sus propias licencias y condiciones de uso.
