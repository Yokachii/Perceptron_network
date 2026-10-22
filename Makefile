cln : perceptron.o main.o dataset.o 
	gcc -g -Wall -Wextra -o perceptron perceptron.o main.o dataset.o -lm

perceptron.o : ./src/perceptron.c ./include/perceptron.h
	gcc -c -g -Wall -Wextra -o perceptron.o ./src/perceptron.c -lm

dataset.o : ./src/dataset.c ./include/dataset.h
	gcc -c -g -Wall -Wextra -o dataset.o ./src/dataset.c -lm

main.o : ./src/main.c 
	gcc -c -g -Wall -Wextra -o main.o ./src/main.c -lm

clean : 
	rm -f *.o
	rm -f perceptron