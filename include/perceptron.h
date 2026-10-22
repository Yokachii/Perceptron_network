#include <stdio.h> 


#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "dataset.h"

typedef struct perceptron_s
{
    int b; // Biais
    int dim; // Dimension
    float taux; // Taux d'apprentissage
    float *poids; // Tableau des wi
} Perceptron;




/**
 * \fn perceptron *new_perceptron(int dim, int taux) 
 * \brief Initialise un perceptron
 *
 * \param dim nombre de wi xi 
 * \param taux taux d'apprentissage 
 * \return pointeur vers le nouveau perceptron
 */ 
Perceptron *new_perceptron(int dim, float taux);


/**
 * \fn void free_perceptron(Perceptron *p)
 * \brief Libère un perceptron
 *
 * \param p Pointeur vers le perceptron à free 
 */ 
void free_perceptron(Perceptron *p);

/**
 * \fn void *display_perceptron(Perceptron p)
 * \brief Affiche un perceptron
 *
 * \param p Perceptron à afficher
 */ 
void display_perceptron(Perceptron p);


/**
 * \fn int prediction(Perceptron p, int* x)
 * \brief Prediction
 *
 * \param p perceptron
 * \param x vecteur
 */ 
float prediction(Perceptron p, float* x);


/**
 * \fn void update(Perceptron *p, float *tab)
 * \brief Met à jour les poids du perceptron
 *
 * \param p perceptron
 * \param tab vecteur de données
 */ 
void update(Perceptron *p, float *tab);


/**
 * \fn int train(Perceptron *p, int nb_epoques, Dataset *d)
 * \brief Entraine le perceptron sur un dataset
 *
 * \param p perceptron
 * \param nb_epoques nombre de cycle
 * \param d Dataset to train on
 */ 
float train(Perceptron *p, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max);


/**
 * \fn float calc_acc(Perceptron *p, Dataset *d)
 * \brief Calcule l'acc d'un perceptron sur un dataset donné
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
float calc_acc(Perceptron *p, Dataset *d);


/**
 * \fn MatrixConfusion matrice_confusion(Perceptron *p, Dataset *d)
 * \brief Calcul la matrix de confusion
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
MatrixConfusion *matrice_confusion(Perceptron *p, Dataset *d);


/**
 * \fn void display_matrix(MatrixConfusion m)
 * \brief Affiche la matrix de confusion
 *
 * \param m La matrix à afficher
 */ 
void display_matrix(MatrixConfusion *m);

#endif