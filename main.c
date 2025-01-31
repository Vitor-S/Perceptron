#include <stdio.h>

// derivada da função de erro em função de W
float calc_derivate(float predict, float y, float x){
    return 2 * (predict - y) * x;
}

// função que calculada a predição
float calc_predict(float x, float w, float b){
    return x * w + b;
}

int main(){
    float x = -10; // entrada em celsius
    float y = 14; // saída esperada em fahrenheit
    float w = 1.6; // valor não ideal
    float b = 32;

    float eta = 0.002; // taxa de aprendizado
    int epochs = 30; // quantidade de vezes que vamos iterar

    for(int e = 0; e < epochs; e++){
        printf("ITERACAO: %i\t W: %f\n", e, w);

        // calcula a saída esperada
        float predict = calc_predict(x, w, b);

        // calcula a inclinação
        float dw = calc_derivate(predict, y, x);

        // reajusta w com base na inclinação (dw)
        w = w - (dw * eta);
    }
    
    return 0;
}