#include "defs.h"

SequenceSet *read_aligned_sequences(char *filename) {
  FILE *fp;
  char line[MAXLINE], *token;
  int len,linecount, i, ok;
  SequenceSet *s;
  char *thisfunction = "read_aligned_sequences";
  
  if ((s = (SequenceSet *) malloc (sizeof(SequenceSet))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc SequenceSet.\n",thisfunction);
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
  s->num_seqs = linecount/2;
  
  if ((s->id = (char **) malloc(s->num_seqs * sizeof(char *))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d char ptr array.\n",thisfunction,s->num_seqs);
    perror("");exit(-1);
  }
  if ((s->seq = (char **) malloc(s->num_seqs * sizeof(char *))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d*%d char array.\n",thisfunction,s->num_seqs,len+1);
    perror("");exit(-1);
  }
  for (i=0; i<s->num_seqs; i++) {
    if ((s->seq[i] = (char *) malloc((len+1) * sizeof(char))) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot malloc %d*%d char array.\n",thisfunction,s->num_seqs,len+1);
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
    
    strncpy(s->seq[i], line, len);
    s->seq[i][len] = '\0';
  }

  fclose(fp);
  
  return(s);
}

double pdist(char *a, char *b, int len) {
  int i,mismatches=0,okpositions=0;

  for (i=0; i<len; i++) {
    /* if (((a[i] == 'A') || (a[i] == 'C') || (a[i] == 'G') || (a[i] == 'T')) || ((b[i] == 'A') || (b[i] == 'C') || (b[i] == 'G') || (b[i] == 'T'))) { */
    if (((a[i] == 'A') || (a[i] == 'C') || (a[i] == 'G') || (a[i] == 'T')) && ((b[i] == 'A') || (b[i] == 'C') || (b[i] == 'G') || (b[i] == 'T'))) { 
      okpositions++;
      if (a[i] != b[i])
	mismatches++;
    }
  }
  if (okpositions)
    return ((double) mismatches/okpositions);
  else
    return (1.0);
}

PtrCountItem *create_ptrcount(int n) {
  PtrCountItem *a;
  if ((a = (PtrCountItem *) malloc (n * sizeof(PtrCountItem))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc %d PtrCountItems.\n",n);
    perror(""); exit(-1);
  }
  return(a);
}

int reset_ptrcount(PtrCountItem *pcount, int n) {
  int i;
  for (i=0; i<n; i++) {
    pcount[i].num_mismatches=0;
    pcount[i].num_notok_positions=0;
  }
  return(0);
}

int *malloc_int_array(int n) {
  int *a;
  if ((a = (int *) malloc (n * sizeof(int))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc %d ints.\n",n);
    perror(""); exit(-1);
  }
  return(a);
}

PtrSeqItem *create_ptrseq(SequenceSet *r) {
  PtrSeqItem *p;
  int i,j, iA,iC,iG,iT,iN;

  if ((p = (PtrSeqItem *) malloc (r->alen * sizeof(PtrSeqItem))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc %d PtrSeqItems.\n",r->alen);
    perror(""); exit(-1);
  }

  for (i=0; i<r->alen; i++) {
    p[i].A=NULL;p[i].C=NULL;p[i].G=NULL;p[i].T=NULL;p[i].N=NULL;
    p[i].numA=0;p[i].numC=0;p[i].numG=0;p[i].numT=0;p[i].numN=0;
    for (j=0; j<r->num_seqs; j++) {
      if (r->seq[j][i] == 'A') p[i].numA++;
      else if (r->seq[j][i] == 'C') p[i].numC++;
      else if (r->seq[j][i] == 'G') p[i].numG++;
      else if (r->seq[j][i] == 'T') p[i].numT++;
      else p[i].numN++;
    }
    if (p[i].numA > 0) p[i].A = malloc_int_array(p[i].numA);
    if (p[i].numC > 0) p[i].C = malloc_int_array(p[i].numC);
    if (p[i].numG > 0) p[i].G = malloc_int_array(p[i].numG);
    if (p[i].numT > 0) p[i].T = malloc_int_array(p[i].numT);
    if (p[i].numN > 0) p[i].N = malloc_int_array(p[i].numN);

    /* put rseq indices to arrays */
    iA=0;iC=0;iG=0;iT=0;iN=0;
    for (j=0; j<r->num_seqs; j++) {
      if (r->seq[j][i] == 'A') {p[i].A[iA]=j; iA++;}
      else if (r->seq[j][i] == 'C') {p[i].C[iC]=j; iC++;}
      else if (r->seq[j][i] == 'G') {p[i].G[iG]=j; iG++;}
      else if (r->seq[j][i] == 'T') {p[i].T[iT]=j; iT++;}
      else {p[i].N[iN]=j; iN++;}
    }
  }
  return(p);
}

int ptrseq_pdist(PtrSeqItem *p, char *s, int alen, PtrCountItem *pcount, int *global_num_notok_positions) {
  int i,j,g=0;

  for (i=0; i<alen; i++) {
    if (s[i] == 'A') {
      for (j=0; j<p[i].numC; j++) pcount[p[i].C[j]].num_mismatches++;
      for (j=0; j<p[i].numG; j++) pcount[p[i].G[j]].num_mismatches++;
      for (j=0; j<p[i].numT; j++) pcount[p[i].T[j]].num_mismatches++;
      for (j=0; j<p[i].numN; j++) pcount[p[i].N[j]].num_notok_positions++;	
    }
    else if (s[i] == 'C') {
      for (j=0; j<p[i].numA; j++) pcount[p[i].A[j]].num_mismatches++;
      for (j=0; j<p[i].numG; j++) pcount[p[i].G[j]].num_mismatches++;
      for (j=0; j<p[i].numT; j++) pcount[p[i].T[j]].num_mismatches++;
      for (j=0; j<p[i].numN; j++) pcount[p[i].N[j]].num_notok_positions++;	
    }
    else if (s[i] == 'G') {
      for (j=0; j<p[i].numA; j++) pcount[p[i].A[j]].num_mismatches++;
      for (j=0; j<p[i].numC; j++) pcount[p[i].C[j]].num_mismatches++;
      for (j=0; j<p[i].numT; j++) pcount[p[i].T[j]].num_mismatches++;
      for (j=0; j<p[i].numN; j++) pcount[p[i].N[j]].num_notok_positions++;	
    }
    else if (s[i] == 'T') {
      for (j=0; j<p[i].numA; j++) pcount[p[i].A[j]].num_mismatches++;
      for (j=0; j<p[i].numC; j++) pcount[p[i].C[j]].num_mismatches++;
      for (j=0; j<p[i].numG; j++) pcount[p[i].G[j]].num_mismatches++;
      for (j=0; j<p[i].numN; j++) pcount[p[i].N[j]].num_notok_positions++;	
    }
    else {
      g++;
    }
  }
  *global_num_notok_positions = g;
  return(0);
}

double get_pdist(PtrCountItem *c, int oklen) {
  return((double) c->num_mismatches / (oklen - c->num_notok_positions));
} 

double get_pdist2(PtrCountItem *c, int oklen, int min_alen) {
  if ((oklen - c->num_notok_positions) >= min_alen)    
    return((double) c->num_mismatches / (oklen - c->num_notok_positions));
  else
    return (1.0);
} 

int print_ptrseq(PtrSeqItem *p, int alen) {
  int i,j;
  for (i=0; i<alen; i++) {
    printf("%d:",i);
    printf(" A%d (",p[i].numA);
    for (j=0; j<p[i].numA; j++) printf(" %d",p[i].A[j]);
    printf(")");

    printf(" C%d (",p[i].numC);
    for (j=0; j<p[i].numC; j++) printf(" %d",p[i].C[j]);
    printf(")");

    printf(" G%d (",p[i].numG);
    for (j=0; j<p[i].numG; j++) printf(" %d",p[i].G[j]);
    printf(")");

    printf(" T%d (",p[i].numT);
    for (j=0; j<p[i].numT; j++) printf(" %d",p[i].T[j]);
    printf(")");

    printf(" A%d (",p[i].numN);
    for (j=0; j<p[i].numN; j++) printf(" %d",p[i].N[j]);
    printf(")\n");
  }
  return(0);
}


