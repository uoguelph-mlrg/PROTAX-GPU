#include <time.h>
#include "defs.h"


double *pdistances;

int compute_distances(SequenceSet *a, char *seq)
{
  int i;
  
  for (i=0; i<a->num_seqs; i++) {
    pdistances[i] = pdist(seq, a->seq[i], a->alen);
  }
  return (0);
}

int compute_cnode_probs_best2(TaxonomyNode *node, int nid, double prevprob, SequenceSet *rseq, Model *m, double **scs, char *seq, double pth) {
  int i,j,cid,k;
  double dist,mindist1, mindist2, maxz,ezsum, *beta, *sc;

  beta = m->params[node[nid].level];
  sc = scs[node[nid].level];
  maxz = 0.0;
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
    node[cid].prob *= prevprob;
    if (node[cid].no_rseqs)
      node[nid].sumcprob_no_rseqs += node[cid].prob;
    if ((node[cid].no_rseqs == 0) && (node[cid].prob >= pth)) {
      printf(" %s %f",node[cid].name, node[cid].prob);
      if (node[cid].num_cnodes)
	compute_cnode_probs_best2(node, cid, node[cid].prob, rseq, m, scs, seq, pth);
    }
  }
  if (node[nid].sumcprob_no_rseqs >= pth) {
    if (node[nid].level == 0)
      printf(" %s %f",UNKNAME, node[nid].sumcprob_no_rseqs);
    else
      printf(" %s,%s %f",node[nid].name, UNKNAME, node[nid].sumcprob_no_rseqs);
  }

  return(0);
}


int main (int argc, char **argv) {
  int i,j, num_tnodes, num_sclevels;
  SequenceSet *rseq,*iseq;
  TaxonomyNode *taxonomy;
  Model *model;
  double pth, **scs;
  time_t start_time, now_time;
  
  if (argc < 7) {
    fprintf(stderr,"usage: classify taxonomy rseqFASTA taxonomy2rseq modelparameters scalingfile probability_threshold inputFASTA\n");
    exit(0);	    
  }

  taxonomy = read_taxonomy(argv[1], &num_tnodes);
  rseq = read_aligned_sequences(argv[2]);
  add_rseq2taxonomy(argv[3], taxonomy);
  model = read_model(argv[4]);
  scs=read_level_scalings(argv[5], &num_sclevels);

  if (model->num_levels != num_sclevels) {
    fprintf(stderr,"ERROR: %d model levels but %d scaling levels, files '%s' and '%s'.\n", model->num_levels, num_sclevels, argv[4], argv[5]);
    exit(0);
  }
  
  pth = atof(argv[6]);  
  iseq = read_aligned_sequences(argv[7]);
  
  if (rseq->alen != iseq->alen) {
    fprintf(stderr,"ERROR: sequence lengths different in two files (%d,%d), files '%s','%s'.\n",rseq->alen,iseq->alen,argv[2],argv[6]);
    exit(0);
  }

  /*
    print_taxonomy(taxonomy, num_tnodes);
    print_model(model);
  */

  if ((pdistances = (double *) malloc(rseq->num_seqs * sizeof(double))) == NULL) {
    fprintf(stderr,"ERROR: cannot maloc %d doubles for pdistances.\n",rseq->num_seqs);
    perror(""); exit(-1);
  }
  
  time(&start_time);
  for (i=0; i<iseq->num_seqs; i++) {
    compute_distances(rseq, iseq->seq[i]);
    printf("%s",iseq->id[i]);
    compute_cnode_probs_best2(taxonomy, 0, 1.0, rseq, model, scs, iseq->seq[i], pth);
    printf("\n");
  }
  time(&now_time);

  printf("time: %f seconds\n",difftime(now_time, start_time));
  
  return(0);
}
