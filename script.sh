gcc -O3 -mavx512f main.c -o main

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force --benchmark
