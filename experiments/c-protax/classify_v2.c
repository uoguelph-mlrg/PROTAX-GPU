#include <time.h>
#include "defs.h"

#define NUCLEOTIDES_IN_WORD 16

typedef struct {
  int num_seqs, alen, ulen, mulen;
  char **id;
  long unsigned int **b, **m;
} SequenceSetB;

int nucleotide2binary(char *s, int n, long unsigned int *b, long unsigned int *m)
{
  long unsigned int a, am;
  int i,j,k, n2, n_remaining;

  /* sequence content, 4 bits for one character */
  
  n2 = n / NUCLEOTIDES_IN_WORD;
  i=0;
  for (j=0; j<n2; j++) {
    a = 0;
    for (k=0; k<NUCLEOTIDES_IN_WORD; k++) {
      a <<= 4;
      if (s[i] == 'A') {a += 1;}
      else if (s[i] == 'C') {a += 2;}
      else if (s[i] == 'G') {a += 4;}
      else if (s[i] == 'T') {a += 8;}
      i++;
    }
    b[j] = a;
  }

  n_remaining = n - n2*NUCLEOTIDES_IN_WORD; 
  if (n_remaining) {
    a = 0;
    for (k=0; k<n_remaining; k++) {
      a <<= 4;
      if (s[i] == 'A') {a += 1;}
      else if (s[i] == 'C') {a += 2;}
      else if (s[i] == 'G') {a += 4;}
      else if (s[i] == 'T') {a += 8;}
      i++;
    }
    b[j] = a;
  }

  /* mask, 1 bit for character: 1 ok, 0 not */
  
  n2 = n / (NUCLEOTIDES_IN_WORD*4);
  i=0;
  for (j=0; j<n2; j++) {
    am = 0;
    for (k=0; k<NUCLEOTIDES_IN_WORD*4; k++) {
      am <<= 1;
      if ((s[i] == 'A') || (s[i] == 'C') || (s[i] == 'G') || (s[i] == 'T')) 
	am += 1;
      i++;
    }
    m[j] = am;
  }

  n_remaining = n - n2*NUCLEOTIDES_IN_WORD*4; 
  if (n_remaining) {
    am = 0;
    for (k=0; k<n_remaining; k++) {
      am <<= 1;
      if ((s[i] == 'A') || (s[i] == 'C') || (s[i] == 'G') || (s[i] == 'T')) 
	am += 1;
      i++;
    }
    m[j] = am;
  }

  return (0);
}


SequenceSetB *read_aligned_sequencesB(char *filename) {
  FILE *fp;
  char line[MAXLINE], *token;
  int len,linecount, i, ok, ulen, mulen;
  SequenceSetB *s;
  char *thisfunction = "read_aligned_sequences";
  
  if ((s = (SequenceSetB *) malloc (sizeof(SequenceSetB))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc SequenceSetB.\n",thisfunction);
    perror(""); exit(-1);
  }
      
  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot open '%s' for reading.\n",thisfunction,filename);
    perror(""); exit(-1);
  }
  
  for (i=0;i<MAXLINE;i++) line[i] = '\0';
  
  linecount=0;
  while (fgets(line, MAXLINE, fp)) {
    linecount++;
    if (line[MAXLINE-2] != '\0') {
      fprintf(stderr,"ERROR (%s): line %d length in file '%s' exceeds MAXLINE %d.\n",thisfunction,linecount,filename,MAXLINE);
      exit(-1);
    }
    if (line[0] != '>') {
      fprintf(stderr,"ERROR (%s): line %d in file '%s' doesn't start with '>' but '%c'.\n",thisfunction,linecount,filename,line[0]);
      exit(-1);
    }
    linecount++;
    if (fgets(line, MAXLINE, fp) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read line %d from file '%s'.\n",thisfunction,linecount,filename);
      exit(-1);
    }
    if (line[MAXLINE-2] != '\0') {
      fprintf(stderr,"ERROR (%s): line %d length in file '%s' exceeds MAXLINE %d.\n",thisfunction,linecount,filename,MAXLINE);
      exit(-1);
    }    
  }
  
  /* calculate sequence length from last sequence */
  len=0;
  while (line[len] != '\n') len++;

  s->alen = len;
  ulen = len / NUCLEOTIDES_IN_WORD;
  if (len > ulen * NUCLEOTIDES_IN_WORD)
    ulen++;
  s->ulen = ulen;
  mulen = len / NUCLEOTIDES_IN_WORD / 4;
  if (len > mulen * NUCLEOTIDES_IN_WORD / 4)
    mulen++;
  s->mulen = mulen;
  s->num_seqs = linecount/2;
  
  if ((s->id = (char **) malloc(s->num_seqs * sizeof(char *))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d char ptr array.\n",thisfunction,s->num_seqs);
    perror("");exit(-1);
  }
  if ((s->b = (long unsigned int **) malloc(s->num_seqs * sizeof(long unsigned int *))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d*%d ul array.\n",thisfunction,s->num_seqs,ulen);
    perror("");exit(-1);
  }
  if ((s->m = (long unsigned int **) malloc(s->num_seqs * sizeof(long unsigned int *))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d*%d ul array.\n",thisfunction,s->num_seqs,ulen);
    perror("");exit(-1);
  }
  for (i=0; i<s->num_seqs; i++) {
    if ((s->b[i] = (long unsigned int *) malloc(ulen * sizeof(long unsigned int))) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot malloc %d*%d ul array.\n",thisfunction,s->num_seqs,ulen);
      perror("");exit(-1);
    }
    if ((s->m[i] = (long unsigned int *) malloc(mulen * sizeof(long unsigned int))) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot malloc %d*%d ul array.\n",thisfunction,s->num_seqs,mulen);
      perror("");exit(-1);
    }
  }
  
  rewind(fp);
  for (i=0;i<s->num_seqs; i++) {
    if (fgets(line, MAXLINE, fp) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read entry %d name (linecount %d) from file '%s'.\n",thisfunction,i,2*i+1,filename);
      perror("");exit(-1);
    }
    token = strtok(line," \t\n");
    s->id[i] = strdup(token+1);
    
    if (fgets(line, MAXLINE, fp) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read entry %d sequence (linecount %d) from file '%s'.\n",thisfunction,i,2*i+2,filename);
      perror("");exit(-1);
    }
    if (strlen(line) != (len+1)) {
      fprintf(stderr,"ERROR (%s): sequence lenghts differ from %d, line %d, file '%s'.\n",thisfunction,len,2*i+2,filename);
      perror("");exit(-1);
    }
    
    nucleotide2binary(line, len, s->b[i], s->m[i]);
  }
  
  fclose(fp);
  
  return(s);
}

double pdistB(long unsigned int *a, long unsigned int *ma, long unsigned int *b, long unsigned int *mb, int n, int n2)
{
  int i, num_ok, num_matches;
  long unsigned int f;
  
  num_ok=0;
  num_matches=0;

  for (i=0; i<n2; i++) {
    num_ok += __builtin_popcountl(ma[i] & mb[i]);
  }

  for (i=0; i<n; i++) {
    num_matches += __builtin_popcountl(a[i] & b[i]);
  }

  if (num_ok > 0)
    return (1.0 - (double) num_matches / num_ok);
  else
    return (1.0);
}


double *pdistances;

int compute_distances(SequenceSetB *a, long unsigned int *b, long unsigned int *m)
{
  int i;
  
  for (i=0; i<a->num_seqs; i++) {
    pdistances[i] = pdistB(b, m, a->b[i], a->m[i], a->ulen, a->mulen);
  }
  return (0);
}

int compute_cnode_probs_best2(TaxonomyNode *node, int nid, double prevprob, Model *m, double **scs, double pth) {
  int i,j,cid,k;
  double dist,mindist1, mindist2, maxz,ezsum, *beta, *sc;

  beta = m->params[node[nid].level];
  sc = scs[node[nid].level];
  maxz = 0.0;

  // compute z for all children
  for (i=0; i<node[nid].num_cnodes; i++) {

    cid = node[nid].cnode_index[i];

    mindist1 = 1.0;
    mindist2 = 1.0;
    for (j=0; j<node[cid].num_rseqs; j++) {
      k = node[cid].rseq_index[j];
      /* dist = pdist(seq, rseq->seq[k], rseq->alen); */
      dist = pdistances[k];
      if (dist < mindist1) {
	mindist2 = mindist1;
	mindist1 = dist;
      }
      else if (dist < mindist2) {
	mindist2 = dist;
      }
    }

    /* printf("  %s %f %f\n",node[cid].name,mindist,avedist); */

    /* use prob temporarily to store z */
    if (node[cid].isunk) {     
      node[cid].prob = 0.0;
      node[cid].no_rseqs = 1;
    }
    else if (node[cid].num_rseqs) {
      if (node[cid].num_rseqs==1) mindist2=mindist1;
      node[cid].prob = beta[1] + beta[2]*(mindist1-sc[0])/sc[1] + beta[3]*(mindist2 - mindist1 - sc[2])/sc[3];
      node[cid].no_rseqs = 0;
    }
    else {
      node[cid].prob = beta[0];
      node[cid].no_rseqs = 1;
    }

    if (node[cid].prob > maxz)
      maxz = node[cid].prob;
  }
  
  // normalizing child probs over parents
  ezsum = 1e-100;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    node[cid].prob = node[cid].prior * exp(node[cid].prob - maxz);
    ezsum += node[cid].prob;
  }

  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    node[cid].prob /= ezsum;
  }

  node[nid].sumcprob_no_rseqs = 0.0;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];

    // node[cid].prob *= prevprob;
    if (node[cid].no_rseqs)
      node[nid].sumcprob_no_rseqs += node[cid].prob;
    
    // only continue traversing nodes WITH refseqs
    if (node[cid].no_rseqs == 0  && (node[cid].prob >= pth)) {
      // print relevant probabilities for nodes with refseqs
      // printf("(%d: %s: %f)", cid, node[cid].name, node[cid].prob);
      if (node[cid].num_cnodes){
        compute_cnode_probs_best2(node, cid, node[cid].prob, m, scs, pth);
      }
	
    }
  }

  // printing unknown node probabilities + taxa with no refseqs(?)
  // if (node[nid].sumcprob_no_rseqs >= pth) {
  //   if (node[nid].level == 0)
  //     printf(": %s: %f", UNKNAME, node[nid].sumcprob_no_rseqs);
  //   else
  //     printf(": %d: %s: %f",nid, UNKNAME, node[nid].sumcprob_no_rseqs);
  // }

  return(0);
}



void print_design_mat(TaxonomyNode *node, int num_nodes, Model *m, double **scs){
  double *beta, *sc, dist; 
  int i,j,k;
  printf("b0, b1, b2, b3\n");

  // root's row, doesn't matter
  printf("1,1,1,1\n");
  double mindist1;
  double mindist2;

  // iterate over all children
  for (i=1; i<num_nodes; i++) {
    // use parent's level
    beta = m->params[node[i].level - 1]; 
    sc = scs[node[i].level - 1];

    mindist1 = 1.0;
    mindist2 = 1.0;

    // iterate over all children's reference sequences
    for (j=0; j<node[i].num_rseqs; j++) {
      k = node[i].rseq_index[j];

      dist = pdistances[k];
      if (dist < mindist1) {
        mindist2 = mindist1;
        mindist1 = dist;
      }
      else if (dist < mindist2) {
        mindist2 = dist;
      }
    }
    
    // printing design matrix
    if (node[i].isunk) {     
      // node[i].prob = 0.0;
      printf("0,0,0,0\n");
    }
    else if (node[i].num_rseqs) {
      if (node[i].num_rseqs==1){
        mindist2=mindist1;
      }
      // node[i].prob = beta[1] + beta[2]*(mindist1-sc[0])/sc[1] + beta[3]*(mindist2 - mindist1 - sc[2])/sc[3];
      printf("0,1,%f,%f\n", (mindist1-sc[0])/sc[1], (mindist2 - mindist1 - sc[2])/sc[3]);
    }
    else {
      // node[i].prob = beta[0];
      // node[i].no_rseqs = 1;
      printf("1,0,0,0\n");
    }
  }
}

void print_model_state(TaxonomyNode *node, int num_nodes){
  /**
   * prints all z within model
  */

  printf("z\n");
  // get number of nodes in taxonomic level
  for (int i=0; i<num_nodes; i++) {
    printf("%.10f\n", node[i].prob);
  }
}


// CUSTOM compare function for quicksort
int compare_nodes(const void *p1, const void *p2){
  TaxonomyNode *a = (TaxonomyNode*)p1;
  TaxonomyNode *b = (TaxonomyNode*)p2;

  if (a->prob < b->prob){
    return 1;
  }
  else if (a->prob > b-> prob){
    return -1;
  }
  else{
    return 0;
  }
}

// CUSTOM compare function for quicksort
int compare_nid(const void *p1, const void *p2){
  TaxonomyNode *a = (TaxonomyNode*)p1;
  TaxonomyNode *b = (TaxonomyNode*)p2;
  if (a->nid < b->nid){
    return -1;
  }
  else if (a->nid > b-> nid){
    return 1;
  }
  else{
    return 0;
  }
}


// CUSTOM FUNCTION: get predictions at one layer
void classify_single(TaxonomyNode *node, int num_nodes, int k, int level){
  /**
   * prints the top k probabilities in the level specified (sorted)
  */

  // get number of nodes in taxonomic level
  int lvl_count = 0;
  for (int i=0; i<num_nodes; i++) {
    if (node[i].level == level){
      lvl_count++;
    }
  }

  // copy nodes over to temporary array
  TaxonomyNode *temp = malloc(lvl_count * sizeof(TaxonomyNode));
  int j = 0;
  for (int i=0; i<num_nodes; i++) {
    if (node[i].level == level && j < lvl_count){
      temp[j] = node[i];
      j++;
    }
  }
  qsort(temp, lvl_count, sizeof(TaxonomyNode), compare_nodes);
  if (lvl_count == 0){
    free(temp);
    return;
  }

  for (int i=0; i<k; i++) {
    if (i >= lvl_count){
      return;
    }
    // printf("%s:%f, ", temp[i].name, temp[i].prob);
    printf(",%d", temp[i].nid);
  }
  free(temp);
}


// CUSTOM FUNCTION: get predictions at layer
// print top k probabilities of each layer
void classify_layer(TaxonomyNode *node, int num_nodes, int k, int level, 
                    SequenceSetB *seqs, SequenceSetB *rseq, Model *m, double **scs){

  printf("root,1,2,3,4,5,6,7\n");
  for (int i=0; i<seqs->num_seqs; i++) {
    for (int j=1; j < 8; j++){
      compute_distances(rseq, seqs->b[i], seqs->m[i]);
      compute_cnode_probs_best2(node, 0, 1.0, m, scs, 0.0);
      classify_single(node, num_nodes, 1, j);
    }
    
    printf("\n");
  }
}


void perplexity_layer(TaxonomyNode *node, int num_nodes, double* perps){
  /**
   * Prints the perplexity of the given layer
  */

  double perps_tmp[8] = {0};

  for (int i=0; i<num_nodes; i++) {
    if (node[i].prob != 0){
      perps_tmp[node[i].level] -= log(node[i].prob);
    }
    
  }

  for (int i=0; i<8; i++){
    perps[i] += exp(perps_tmp[i]/num_nodes);
  }

}


int main (int argc, char **argv) {
  int i,j, num_tnodes, num_sclevels;
  SequenceSetB *rseq,*iseq;
  TaxonomyNode *taxonomy;
  Model *model;
  double pth, **scs;
  clock_t start_time, now_time;

  // arguments for evaluation
  int topk;
  int classify_lvl;

  
  if (argc < 9) {
    fprintf(stderr,"usage: classify taxonomy rseqFASTA taxonomy2rseq modelparameters scalingfile probability_threshold inputFASTA\n");
    exit(0);	    
  }

  taxonomy = read_taxonomy(argv[1], &num_tnodes);
  rseq = read_aligned_sequencesB(argv[2]);
  add_rseq2taxonomy(argv[3], taxonomy);
  model = read_model(argv[4]);
  scs=read_level_scalings(argv[5], &num_sclevels);
  
  topk = atoi(argv[8]);

  if (model->num_levels != num_sclevels) {
    fprintf(stderr,"ERROR: %d model levels but %d scaling levels, files '%s' and '%s'.\n", model->num_levels, num_sclevels, argv[4], argv[5]);
    exit(0);
  }
  
  pth = atof(argv[6]);  // threshold probabilities
  iseq = read_aligned_sequencesB(argv[7]);
  
  if (rseq->alen != iseq->alen) {
    fprintf(stderr,"ERROR: sequence lengths different in two files (%d,%d), files '%s','%s'.\n",rseq->alen,iseq->alen,argv[2],argv[6]);
    exit(0);
  }

  if ((pdistances = (double *) malloc(rseq->num_seqs * sizeof(double))) == NULL) {
    fprintf(stderr,"ERROR: cannot maloc %d doubles for pdistances.\n",rseq->num_seqs);
    perror(""); exit(-1);
  }

  // inference time
  double inf_time = 0;

  // perplexity
  double perps[8] = {0};

  classify_layer(taxonomy, num_tnodes, 1, j, iseq, rseq, model, scs);
  return(0);
}
