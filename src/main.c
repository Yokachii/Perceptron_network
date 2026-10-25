#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {

    if (argc<2) {
        fprintf(stderr, "Usage: %s <dataset_path> <nb_epoques> <learning_rate> <train_ratio> <normalize> <StopWhenTrained> <HumanOutput> <nb_neurones> <nb_couches_cachees> <graine>\n", argv[0]);
        fprintf(stderr, "Example: %s ./datasets/cercles.csv 1000 0.1 0.8 1 0 1 32 2 42\n", argv[0]);
        return 1;
    }

    char *filepath = argv[1];

    int nb_epoques = (argc > 2) ? atoi(argv[2]) : 1000;
    float taux = (argc > 3) ? atof(argv[3]) : 1;
    float train_ratio = (argc > 4) ? atof(argv[4]) : 0.8;
    int normalizeArg = (argc > 5) ? atoi(argv[5]) : 1;
    int StopWhenTrained = (argc > 6) ? atoi(argv[6]) : 0;
    int HumanOutput = (argc > 7) ? atoi(argv[7]) : 1;
    int NbPerceptrons = (argc > 8) ? atoi(argv[8]) : 32;
    int NbCouches = (argc > 9) ? atoi(argv[9]) : 2;
    unsigned int seed = (argc > 10) ? (unsigned int)strtoul(argv[10], NULL, 10) : (unsigned int)time(NULL);

    srand(seed);   // avant new_network, qui tire les poids initiaux
    if(HumanOutput==1) printf("Graine : %u\n", seed);

    Dataset *d = load_dataset(filepath);
    Reseau *n = new_network(NbCouches, NbPerceptrons, taux, d->dim - 1, 1);
    Dataset *tr, *tt;
    shuffle(d);
    if(normalizeArg==1){
        normalize(d);
    }
    split_dataset(d,&tr,&tt,train_ratio);

    int isFreed = 0;

    if(tt->n==0){
        free_dataset(tt);
        tt=tr;
        isFreed=1;
    }

    float accBefore=calc_acc(n,tt);
    if(HumanOutput==1) printf("\nAcc avant train : %f\n\n",accBefore);

    int epoquesMax;
    float maxAcc;
    int epoques = train(n,nb_epoques,tr,tt,StopWhenTrained,&epoquesMax,&maxAcc,HumanOutput);

    MatrixConfusion *matrix = matrice_confusion(n,tt);
    float accAfter=calc_acc(n,tt);

    if(HumanOutput==1){
        display_network(*n);
        printf("Acc après train : %f\n\n",accAfter);
        printf("epoquesMax : %d\nAccMax:%f\n",epoquesMax,maxAcc);
        display_matrix(matrix);
    }else{
        // #END;epoques;accBefore;accAfter;FN;FP;TN;TP;epoquesMax;accMax
        printf("#END;%d;%f;%f;%d;%d;%d;%d;%d;%f\n",epoques,accBefore,accAfter,matrix->FN,matrix->FP,matrix->TN,matrix->TP,epoquesMax,maxAcc);
    }

    free(matrix);

    if(isFreed==0){
        free_dataset(tt);
    }

    free_network(n);
    free_dataset(tr);
    return 0;
}