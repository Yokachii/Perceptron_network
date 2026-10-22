#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {

    if (argc<2) {
        fprintf(stderr, "Usage: %s <dataset_path> <nb_epoques (Optionel)> <learning_rate (Optionel)> <train_ratio (Optional)> <normalize (Optional)> <StopWhenTrained> <HumanOutput>\n", argv[0]);
        fprintf(stderr, "Example: %s ./data/champignons.csv 1000 0.1 1 0\n", argv[0]);
        return 1;
    }

    char *filepath = argv[1];

    int nb_epoques = (argc > 2) ? atoi(argv[2]) : 1000;
    float taux = (argc > 3) ? atof(argv[3]) : 0.1;
    float train_ratio = (argc > 4) ? atof(argv[4]) : 1.0;
    int normalizeArg = (argc > 5) ? atoi(argv[5]) : 0;
    int StopWhenTrained = (argc > 6) ? atoi(argv[6]) : 0;
    int HumanOutput = (argc > 7) ? atoi(argv[7]) : 1;

    Dataset *d = load_dataset(filepath);
    Perceptron *p = new_perceptron(d->dim,taux);
    Dataset *tr, *tt;
    split_dataset(d,&tr,&tt,train_ratio);
    
    int isFreed = 0;
    
    if(tt->n==0){
        free_dataset(tt);
        tt=tr;
        isFreed=1;
    }
    
    if(normalizeArg==1){
        normalize(tr);
        normalize(tt);
    }

    float accBefore=calc_acc(p,tt);
    if(HumanOutput==1) printf("\nAcc avant train : %f\n\n",accBefore);

    int epoquesMax;
    float maxAcc;
    int epoques = train(p,nb_epoques,tr,tt,StopWhenTrained,&epoquesMax,&maxAcc);

    MatrixConfusion *matrix = matrice_confusion(p,tt);
    

    float accAfter=calc_acc(p,tt);
    if(HumanOutput==1){
        printf("Acc après train : %f\n\n",accAfter);
        printf("epoquesMax : %d\nAccMax:%f\n",epoquesMax,maxAcc);
        display_matrix(matrix);
    }else{
        printf("%d;%f;%f;%d;%d;%d;%d;%d;%f",epoques,accBefore,accAfter,matrix->FN,matrix->FP,matrix->TN,matrix->TP,epoquesMax,maxAcc);
    }



    if(isFreed==0){
        free_dataset(tt);
    }

    free(matrix);
    free_dataset(tr);
    free_perceptron(p);
}