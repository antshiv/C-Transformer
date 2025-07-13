gcc -O3 -march=native -mavx512f -fopenmp main.c -o main -lm

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force --benchmark
