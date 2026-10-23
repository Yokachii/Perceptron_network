#include "../include/perceptron.h"
#include <stdlib.h>

Reseau *new_network(int nb_couche_cache, int nb_neuronnes_ch, float taux, int dimE, int dimO){
    Reseau *res = malloc(sizeof(Reseau));

    res->nb_layer=2+nb_couche_cache;
    
    Couche **layers = malloc(sizeof(Couche*)*(res->nb_layer));

    int dimL0=dimE;

    Couche *entree = new_layer(dimE,taux,dimL0);

    layers[0] = entree;

    for(int i = 0; i<nb_couche_cache;i++){
        int dimL = nb_neuronnes_ch;
        if(i==0) dimL=dimL0;
        printf("DimL : %d, I : %d\n",dimL,i);
        Couche *l = new_layer(dimL,taux,nb_neuronnes_ch);
        layers[i+1] = l;
    }

    Couche *out = new_layer(nb_neuronnes_ch,taux,dimO);

    layers[res->nb_layer-1] = out;

    res->Layers=layers;

    return res;
}

void display_network(Reseau n){
    printf("Taux : %.2f\n",n.Layers[0]->taux);
    printf("\nCouche d'entré :\n");
    display_perceptron(*(n.Layers[0]));
    printf("\n\n");
    for(int i = 0; i<n.nb_layer-2;i++){
        printf("Couche N°%d (de %d neuronnes):\n",i+1,n.Layers[0]->nb_neuronnes);
        display_perceptron(*(n.Layers[i+1]));
        printf("\n\n");
    }
    printf("Couche de sortie :\n");
    display_perceptron(*(n.Layers[n.nb_layer-1]));
    
}

void free_network(Reseau *n){
    for(int i = 0;i<n->nb_layer;i++){
        free_layer(n->Layers[i]);
    }
    free(n);
}

/**
 * \fn int new_perceptron(int dim, int taux) 
 * \brief Initialise un perceptron
 *
 * \param dim nombre de wi xi 
 * \param n nombre de données 
 * \return pointeur vers le nouveau perceptron
 */ 
Couche *new_layer(int dim, float taux, int nb_neuronnes){
    Couche *n_perceptron = malloc(sizeof(Couche));

    n_perceptron->b = 0;
    n_perceptron->dim = dim;
    n_perceptron->taux = taux;
    n_perceptron->nb_neuronnes = nb_neuronnes;

    float *biais = malloc(sizeof(float)*nb_neuronnes);

    for(int i = 0; i<nb_neuronnes; i++){
        biais[i]=0;
    }

    n_perceptron->b=biais;


    float **poids = malloc(sizeof(float*)*nb_neuronnes);

    for(int i = 0; i<nb_neuronnes; i++){
        float *p = malloc(sizeof(float)*dim);

        for(int j = 0; j<dim; j++){
            p[j]=0;
        }

        poids[i]=p;
    }

    n_perceptron->poids=poids;

    return n_perceptron;
};


/**
 * \fn void free_layer(Perceptron *p)
 * \brief Libère un perceptron
 *
 * \param p Pointeur vers le perceptron à free 
 */ 
void free_layer(Couche *p){
    int nb_neuronnes = p->nb_neuronnes;
    for(int i = 0;i<nb_neuronnes;i++){
        free(p->poids[i]);
    }
    free(p->b);
    free(p);
};

/**
 * \fn void *display_perceptron(Perceptron p)
 * \brief Affiche un perceptron
 *
 * \param p Perceptron à afficher
 */ 
void display_perceptron(Couche p){
    printf("Dim : %d\n",p.dim);
    for(int j = 0; j<p.nb_neuronnes;j++){
        float b = p.b[j];
        int dim = p.dim;
        printf("Perceptron N°%d\nBiais : %.2f\n[\n    ",j+1,b);
        for(int i = 0; i<dim; i++){
            printf("%.2f, ",p.poids[j][i]);
        }
        printf("\n]\n\n");
    }
};

/**
 * \fn float* prediction(Perceptron p, float* x)
 * \brief Prediction -> calc `z = w·x + b` puis retourne `σ(z)`
 *
 * \param p perceptron
 * \param x vecteur
 */ 
float* prediction(Couche p, float* x) {
    float *outputs = malloc(sizeof(float) * p.nb_neuronnes);

    for (int j = 0; j < p.nb_neuronnes; j++) {
        float z = p.b[j];
        
        for (int i = 0; i < p.dim; i++) {
            z += x[i] * p.poids[j][i];
        }
        
        outputs[j] = sigmoide(z);
    }

    return outputs;
}

/**
 * \fn void update(Perceptron *p, float *tab)
 * \brief Met à jour les poids du perceptron
 *
 * \param p perceptron
 * \param tab vecteur de données
 */ 
void update(Couche *p, float *tab) {
    int dim = p->dim;
    
    float *y = prediction(*p, tab);

    for (int j = 0; j < p->nb_neuronnes; j++) {
        float label = tab[dim - 1];
        
        float error = label - y[j];

        for (int i = 0; i < dim; i++) {
            p->poids[j][i] += (p->taux) * error * tab[i];
        }

        p->b[j] += (p->taux) * error;
    }

    free(y);
}


/**
 * \fn void train(Perceptron *p, int nb_epoques, Dataset *d)
 * \brief Entraine le perceptron sur un dataset
 *
 * \param p perceptron
 * \param nb_epoques nombre de cycle
 * \param d Dataset to train on
 */ 
float train(Couche *p, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max){
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
float calc_acc(Couche *p, Dataset *d){
    int n = d->n;

    if(d->dim != p->dim) {
        perror("Dimension missmatch");
        return -1;
    }

    float res = 0;

    for(int i = 0; i < n; i++){
        float *y_array = prediction(*p, d->data[i]);
        
        float y = y_array[p->nb_neuronnes - 1]; 
        
        float label = d->data[i][d->dim - 1];
        
        if((y >= 0.5 && label >= 0.5) || (y < 0.5 && label < 0.5)) {
            res += 1;
        }

        free(y_array);
    }

    return res / n;
}



/**
 * \fn MatrixConfusion matrice_confusion(Perceptron *p, Dataset *d)
 * \brief Calcul la matrix de confusion
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
/**
 * \fn MatrixConfusion *matrice_confusion(Perceptron *p, Dataset *d)
 * \brief Calcule la matrice de confusion
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
MatrixConfusion *matrice_confusion(Couche *p, Dataset *d){

    int n = d->n;
    int dim = p->dim;
    
    MatrixConfusion *res = malloc(sizeof(MatrixConfusion));
    if (res == NULL) return NULL;

    res->FN = 0;
    res->TN = 0;
    res->FP = 0;
    res->TP = 0;

    for(int i = 0; i < n; i++){
        float *y_array = prediction(*p, d->data[i]);
        
        float y = y_array[p->nb_neuronnes - 1]; 
        
        float label = d->data[i][dim - 1];
        
        if(y >= 0.5 && label >= 0.5) res->TP += 1;
        else if(y >= 0.5 && label < 0.5) res->FP += 1;
        else if(y < 0.5 && label < 0.5) res->TN += 1;
        else if(y < 0.5 && label >= 0.5) res->FN += 1;

        free(y_array);
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