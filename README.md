# Projeto TextLab
## 1. Objetivo
Evaluation of text with NeSy

# Excecução Docker
## 1. Criar Imagem a partir de um arquivo dockerfile
```
docker build -t cfabianbp20/docker-scallop:latest .
```
## 2. Listar as imagens
```
docker images
```
## 3. Criar um contenedor
```
docker run -d --gpus all -it --name scallop_gpu cfabianbp20/docker-scallop:latest
```
### Criar um contenedor com um volumem
De jeito local:
Para ver la ruta: $PWD
```
docker run -d --gpus all --name scallop_gpu -v "/home/usuario/Documents/ENIAC - Artigo/lab/data:/home/scallop_user/labs/data" -v "/home/usuario/Documents/ENIAC - Artigo/lab/eniac/result:/home/scallop_user/labs/nyse/result" cfabianbp20/docker-scallop:latest
```
No servidor:
```
docker run -d --gpus all --name scallop_gpu -v "/media/data/christian/git_pry/lab/data:/home/scallop_user/labs/data" -v "/media/data/christian/git_pry/lab/eniac/result:/home/scallop_user/labs/nyse/result" cfabianbp20/docker-scallop:latest
```
```
docker run -d --gpus all --name scallop_gpu -v "/media/data/christian/git_pry/lab/data:/home/scallop_user/labs/data" -v "/media/data/christian/git_pry/lab/LogicProgram-Essay-Unpretrained/result:/home/scallop_user/labs/nyse/result" cfabianbp20/docker-scallop:latest
```
## 4. Listar os contems
```
docker ps -a
```
## 5. Ingresar o bash
```
docker exec -it scallop_gpu bash
```
## 6. Eliminar
### Contem
```
docker rm cfabianbp20/docker-scallop:latest
```
### Imagem 
```
docker rmi scallop_gpu
```
## 7. Permisão de escritura por usuario docker
```
docker exec -it scallop_gpu bash
id
```
Depois
```
sudo chown -R 1000:1000 "/home/usuario/Documents/ENIAC - Artigo/lab/eniac/result"
```
## 7. Actualizar pacotes
```
pip install -U transformers accelerate
```
## 8. Execução em segundo plano
```
nohup python qwen.py > saida.log 2>&1 &
```
Mostrar
```
tail -f saida.log
```
Para interromper
```
pkill -f qwen.py
```
Para verificar se o processo foi interrompido corretamente, temos uma declaração que indica que o processo não foi finalizado; caso contrário, nada é exibido, ou seja, o processo foi finalizado corretamente.
```
ps aux | grep '[q]wen.py'
```
```
pgrep -af qwen.py
```