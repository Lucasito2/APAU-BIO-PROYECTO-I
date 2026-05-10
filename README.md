# Acoplamiento Bioinspirado: REINFORCE + STDP

Este repositorio contiene la implementación empírica de un sistema híbrido que sustituye el optimizador de gradiente tradicional (Adam) del algoritmo REINFORCE por una regla local bioinspirada (Plasticidad Dependiente del Tiempo de Espiga - STDP) modulada por recompensa. 

Este proyecto evalúa si una regla de aprendizaje local de tres factores puede sustituir la necesidad de un optimizador global en el entorno `CartPole-v1`.

---

## (a) Instrucciones de Instalación

Para garantizar la reproducibilidad del entorno, se recomienda utilizar Python 3.9 o superior y crear un entorno virtual. Ejecuta los siguientes comandos en tu terminal:

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   cd <NOMBRE_DE_LA_CARPETA>

2. **Crear y activar el entorno virtual:**
   * En Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   * En macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instalar las dependencias requeridas:**
   ```bash
   pip install -r requirements.txt