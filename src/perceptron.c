#include "../include/perceptron.h"
#include <stdlib.h>

/**
 * \fn int new_perceptron(int dim, int taux) 
 * \brief Initialise un perceptron
 *
 * \param dim nombre de wi xi 
 * \param n nombre de données 
 * \return pointeur vers le nouveau perceptron
 */ 
Perceptron *new_perceptron(int dim, float taux){
    Perceptron *n_perceptron = malloc(sizeof(Perceptron));

    n_perceptron->b = 0;
    n_perceptron->dim = dim;
    n_perceptron->taux = taux;

    float *poids = malloc(sizeof(float)*dim);

    for(int i = 0; i<dim; i++){
        poids[i]=0;
    }

    n_perceptron->poids=poids;

    return n_perceptron;
};


/**
 * \fn void free_perceptron(Perceptron *p)
 * \brief Libère un perceptron
 *
 * \param p Pointeur vers le perceptron à free 
 */ 
void free_perceptron(Perceptron *p){
    free(p->poids);
    free(p);
};

/**
 * \fn void *display_perceptron(Perceptron p)
 * \brief Affiche un perceptron
 *
 * \param p Perceptron à afficher
 */ 
void display_perceptron(Perceptron p){
    int b = p.b;
    float taux = p.taux;
    int dim = p.dim;
    printf("Taux : %.2f\nBiais : %d\nDim : %d\n[\n",taux,b,dim);
    for(int i = 0; i<dim; i++){
        printf("%.2f, ",p.poids[i]);
    }
    printf("\n]\n\n");
};


/**
 * \fn int prediction(Perceptron p, int* x)
 * \brief Prediction -> calc `z = w·x + b` puis retourne `σ(z)`
 *
 * \param p perceptron
 * \param x vecteur
 */ 
float prediction(Perceptron p, float* x){
    int dim=p.dim;
    float res = 0;
    for(int i = 0; i<dim-1; i++){
        float Wi = p.poids[i];
        float Xi = x[i];
        res += Xi*Wi;
    }
    float z =res + p.b;
    return sigmoide(z);
}

/**
 * \fn void update(Perceptron *p, float *tab)
 * \brief Met à jour les poids du perceptron
 *
 * \param p perceptron
 * \param tab vecteur de données
 */ 
void update(Perceptron *p, float *tab){
    int dim=p->dim;
    int label = tab[dim-1];
    float y = prediction(*p,tab);
    for(int i = 0; i<dim-1; i++){
        float Wi = p->poids[i];
        float Xi = tab[i];
        p->poids[i] = Wi + (p->taux)*(label - y)*Xi;
    }
    p->b = p->b + (p->taux)*(label - y);
}



/**
 * \fn void train(Perceptron *p, int nb_epoques, Dataset *d)
 * \brief Entraine le perceptron sur un dataset
 *
 * \param p perceptron
 * \param nb_epoques nombre de cycle
 * \param d Dataset to train on
 */ 
float train(Perceptron *p, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max){
    int firstEpoqueInt;
    float firstEpoqueV;

    int bestAccEp;
    float bestAcc;
    for(int i = 0; i<nb_epoques; i++){
        shuffle(d);

        float acc = calc_acc(p,dt);

        if(breakIfTrained){
            // printf("Acc sur dataset de test après %d epoques : %f\n",i,acc);

            if(acc==1) {
                *acc_max=1;
                *nb_epoques_max=acc;
                return i;
            }
        }

        if(firstEpoqueV!=acc){
            firstEpoqueV=acc;
            firstEpoqueInt=i;
        }

        if(acc>bestAcc){
            bestAcc=acc;
            bestAccEp=i;
        }

        for(int j = 0; j<d->n; j++){
            update(p,d->data[j]);
        }
    }

    *acc_max=bestAcc;
    *nb_epoques_max=bestAccEp;

    return firstEpoqueInt;
}



/**
 * \fn float calc_acc(Perceptron *p, Dataset *d)
 * \brief Calcule l'acc d'un perceptron sur un dataset donné
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
float calc_acc(Perceptron *p, Dataset *d){
    int n = d->n;

    if(d->dim != p->dim) {
        perror("Dimension missmatch");
        return -1;
    }

    float res = 0;

    for(int i = 0; i<n; i++){
        float y = prediction(*p,d->data[i]);
        float label = d->data[i][d->dim-1];
        if((y>=0.5 && label>=0.5) || (y<0.5 && label<0.5)) res+=1;
    }

    return res/n;
}



/**
 * \fn MatrixConfusion matrice_confusion(Perceptron *p, Dataset *d)
 * \brief Calcul la matrix de confusion
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
MatrixConfusion *matrice_confusion(Perceptron *p, Dataset *d){

    int n = d->n;
    int dim =p->dim;
    
    MatrixConfusion *res = malloc(sizeof(MatrixConfusion));

    res->FN=0;
    res->TN=0;
    res->FP=0;
    res->TP=0;


    for(int i = 0; i<n; i++){
        float y = prediction(*p,d->data[i]);
        float label = d->data[i][dim-1];
        if(y>=0.5 && label>=0.5) res->TP+=1;
        if(y>=0.5 && label<0.5) res->FP+=1;
        if(y<0.5 && label<0.5) res->TN+=1;
        if(y<0.5 && label>=0.5) res->FN+=1;
    }

    return res;
    
}



/**
 * \fn void display_matrix(MatrixConfusion m)
 * \brief Affiche la matrix de confusion
 *
 * \param m La matrix à afficher
 */ 
void display_matrix(MatrixConfusion *m){

    printf("FP : %d\nFN : %d\nTP : %d\nTN : %d\n\n",m->FP,m->FN,m->TP,m->TN);
    
}