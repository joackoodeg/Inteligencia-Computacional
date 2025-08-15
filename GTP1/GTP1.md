```
UNL - FICH - Departamento de Inform ́atica - Ingenier ́ıa Inform ́atica
```
## Inteligencia Computacional

```
Gu ́ıa de trabajos pr ́acticos 1
```
# Perceptr ́on simple

## Trabajos pr ́acticos

Ejercicio 1: Implemente rutinas que permitan el entrenamiento y prueba de un
perceptr ́on simple con una cantidad variable de entradas. Se deben tener en
cuenta las siguientes capacidades:

```
lectura de los patrones de entrenamiento (entradas y salidas) desde un
archivo en formato texto separado por comas,
selecci ́on del criterio de finalizaci ́on del entrenamiento y el n ́umero m ́axi-
mo de ́epocas,
selecci ́on de la tasa de aprendizaje,
prueba del perceptr ́on entrenado mediante archivos de texto con el mismo
formato separado por comas.
```
```
Una vez obtenido dicho programa, pru ́ebelo en la resoluci ́on del problema OR,
utilizando los archivos de patronesORtrn.csvyORtst.csvpara el entre-
namiento y la prueba, respectivamente. Los patrones que se proveen en estos
archivos fueron generados a partir de los puntos (1,1), (1,-1), (-1,1) y (-1,-1)
con peque ̃nas desviaciones aleatorias (<5 %) en torno a ́estos. Recuerde que
para que la prueba tenga validez se deben utilizar patrones nunca presentados
en el entrenamiento, para ́esto se dispone de dos archivos de datos diferentes.
```
Ejercicio 2: Implemente una rutina de graficaci ́on que permita visualizar, para el
caso de dos entradas, los patrones utilizados y la recta de separaci ́on que se va
ajustando durante el entrenamiento del perceptr ́on simple. Utilice dicha rutina
para visualizar el entrenamiento en los problemas OR y XOR (utilizando los
archivos de datosORtrn.csvyXORtrn.csv).

Ejercicio 3: Repita el entrenamiento realizado para el caso del OR, pero entrenan-
do con los archivosOR 50 trn.csvyOR 90 trn.csv, y sus correspondientes
OR 50 tst.csvyOR 90 tst.csvpara test. Estos datos fueron generados de la
misma forma que los usados en el primer ejercicio, pero utilizando desviaciones
aleatorias de 50 % y 90 %, respectivamente. Analice y discuta los resultados.

### 1


