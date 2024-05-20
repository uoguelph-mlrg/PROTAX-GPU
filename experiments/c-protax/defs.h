#ifndef _DEFS_
#define _DEFS_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAXLINE 1024
#define UNKNAME "unk"

typedef struct {
  int nid, pid, level;
  char *name, *mxcode;
  int num_cnodes;
  int *cnode_index;
  int num_rseqs;
  int *rseq_index;
  int isunk, no_rseqs;
  double prior, prob, sumcprob_no_rseqs;
} TaxonomyNode;

typedef struct {
  int num_seqs, alen;
  char **seq, **id;
} SequenceSet;

typedef struct {
  int *A,numA,*C,numC,*G,numG,*T,numT,*N,numN;
} PtrSeqItem;

typedef struct {
  int num_mismatches;
  int num_notok_positions;
} PtrCountItem;

typedef struct {
  int num_levels, dim;
  double **params;
} Model;

/* routines_taxonomy.c */

TaxonomyNode *read_taxonomy(char *filename, int *num_nodes);
int add_rseq2taxonomy(char *filename, TaxonomyNode *node);
int print_taxonomy(TaxonomyNode *node, int num_nodes);

/* routines_sequence.c */

SequenceSet *read_aligned_sequences(char *filename);
double pdist(char *a, char *b, int len);
PtrSeqItem *create_ptrseq(SequenceSet *r);
PtrCountItem *create_ptrcount(int n);
int reset_ptrcount(PtrCountItem *pcount, int n);
int ptrseq_pdist(PtrSeqItem *p, char *s, int alen, PtrCountItem *pcount, int *global_num_notok_positions);
double get_pdist(PtrCountItem *c, int oklen);
double get_pdist2(PtrCountItem *c, int oklen, int min_alen);
int print_ptrseq(PtrSeqItem *p, int alen);
int *malloc_int_array(int n);

/* routines_model.c */

Model *read_model(char *filename);
double **read_level_scalings(char *filename, int *num_levels);
int compute_cnode_probs(TaxonomyNode *node, int nid, double prevprob, SequenceSet *rseq, Model *m, char *seq, double pth);
int print_model(Model *m);

#endif
