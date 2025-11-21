gcc -O3 -march=native -mavx512f -fopenmp main.c -o main -lm

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force

./main --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force --benchmark

./main --layers 10 --dmodel 8192 --ctx 4096 --vocab 50000 --head-dim 96 --force --benchmark

 gcc -O3 -march=native -mavx512f -fopenmp main.c -o main -lm
 ./main --weights gpt2_bump.weights --force

 python3 run.py "What is the capital of France?"
 python encode.py "What is newtons 2nd law?"
 ./main --weights gpt2_bump.weights --prompt "2061,318,649,27288,362,358,1099,30" --force

./validate_all.sh "Hello World" gpt2_bump.weights ./main 20 0

  python3 validate_layer_stages.py "Hello World" \
    --layer 0 \
    --weights gpt2_bump.weights \
    --executable ./main


