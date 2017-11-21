# Argument mining with BiLSTM
Implementación de una red BiLSTM-CNN-CRF usada para Sequence Tagging. Adaptación para Argument Mining. Flexibilización de input y reporte de hiperparametros.

## Objetivos
Basandonos en los repositorios de [UKPLab](https://github.com/UKPLab):

- https://github.com/UKPLab/acl2017-neural_end2end_am
- https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf

Adaptar el código para..
- Desarrollar un entorno para la identificación y etiquetado de argumentos en textos del idioma inglés.
- Proponer alternativas de embeddings distintos para entrenamiento/evaluación.
- Flexibilizar input para formato texto o formato CoNLL.
- Reportar resultados con diversos parámetros.

## Arquitectura
![Arquitectura de la red](arch.jpeg)


## Instrucciones para su uso
### Kickstart

```
$ source kickstart.sh
```
Esto creará un entorno virtual dentro de la carpeta ```.env```, instalará las librerías necesarias y se crearán las carpetas vacias ```tmp/``` donde se almacenarán los etiquetados durante las epocas de entrenamiento con el formato: ```[]_[].txt``` , y  ```pkl/```, lugar donde se guardará el preproceso con el embedding seleccionado sobre el corpus de entrenamiento con el formato: ```[dataset_name]_[embedding_filename].pkl```.


### Archivos de entrenamiento
Se proponen 2 tipos de datasets. 
- Uno simple con las categorias de etiquetado básico tales como **Premise**, **Claim**, **MajorClaim**, **O** sin distinguir el caso de que sea a favor o en contra en el caso de las premisas y las afirmaciones. Este dataset está disponible en ```/data/am_simplest/```. En el caso de querer reetiquetar en este formato un essay en formato CoNLL usar el script [util/tag_simplifier.py](util/tag_simplifier.py)

- Otro completo con la adición de las etiquetas **:For** y **:Against** sobre las ya prpuestas. Este dataset se encuentra dentro de ```/data/am/```

Ambos dataset estas listos y dispuestos en formato CoNLL para ser entrenados. **dev.txt, train.txt** para entrenar la red y **test.txt** para probar los resultados.

### Mas archivos de evaluación
Si se necesitaran mas ejemplos de archivos pueden obtenerse [aqui](https://www.ukp.tudarmstadt.de/fileadmin/user_upload/Group_UKP/data/argument-recognition/ArgumentAnnotatedEssays-1.0.zip)

O correr el script siguiente y los tendrán disponibles en **example_essays** en su formato *.txt* para ser evaluados y comparados con su archivo etiquetado *.ann*
```
$ download_examples
```

### Obtener modelo entrenado.
Para entrenar un modelo correr el siguiente comando
```
$ python Train_AM.py [dataset] [embedding] [opts]
```
donde las opciones para cada parametro se detallan debajo.

- **dataset**: am, am_simplest
- **embedding**: levy, word2vec, glove
- **[opts]**: --optimizer, --classifier, --cnn , --help

TODO: marcar como se guardan los modelos entrenados y cuando corta el entrenamiento(>5 epochs without changes).

### Etiquetar texto con modelo entrenado

Para etiquetar texto ajeno con un modelo preentrenado ejecutar el comando RunModel de la siguiente forma.
```
$ python RunModel /models/[dataset]/AM_TAG/[selectedModel].h5 [input.txt]
```
El etiquetado se imprimirá por stdout, aunque si un archivo de salida fuera necesario es posbile su redirección concatenando ```> [output_file]```

### Evaluar eficiencia
TODO: generar eval.py que evalue los output etiquetados comparandolos con los reales. True vs predicted y reporte los errores

## Adaptacion de input:
TODO: Generar en RunModel si pasan un texto en connl el raw para ingresarlo a la red

## Análisis de resultados
TODO
- Con simple tagging
	2 embeddings
	3 optimizers
- full taggin
    same 2 embeddings 3 optimizers
    con cnn
    con cnn y crf

