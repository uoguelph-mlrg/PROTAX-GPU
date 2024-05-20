#include "defs.h"

TaxonomyNode *read_taxonomy(char *filename, int *num_nodes) {
  FILE *fp;
  char line[MAXLINE], *token;
  int linecount, i,j;
  TaxonomyNode *node;
  char *thisfunction = "read_taxonomy_file";
  
  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot open '%s' for reading.\n",thisfunction,filename);
    perror(""); exit(-1);
  }

  linecount=0;
  while (fgets(line, MAXLINE, fp)) {
    linecount++;
  }
  *num_nodes = linecount;

  if ((node = (TaxonomyNode *) malloc ((*num_nodes) * sizeof(TaxonomyNode))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d TaxonomyNodes, file '%s'.\n",thisfunction,*num_nodes, filename);
    perror("");exit(-1);
  }

  rewind(fp);
  linecount=0;
  while (fgets(line, MAXLINE, fp)) {
    if ((token = strtok(line," \t")) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read first token of line %d, file '%s'.\n",thisfunction,linecount+1,filename);
      perror("");exit(-1);
    }
    node[linecount].nid = atoi(token);
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read 2nd token of line %d, file '%s'.\n",thisfunction,linecount+1,filename);
      perror("");exit(-1);
    }
    node[linecount].pid = atoi(token);
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read 3rd token of line %d, file '%s'.\n",thisfunction,linecount+1,filename);
      perror("");exit(-1);
    }
    node[linecount].level = atoi(token);
    if ((token = strtok(NULL," \t\n")) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read 4th token of line %d, file '%s'.\n",thisfunction,linecount+1,filename);
      perror("");exit(-1);
    }
    node[linecount].name = strdup(token);

    if ((token = strtok(NULL," \t\n")) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot read 5th token (prior) of line %d, file '%s'.\n",thisfunction,linecount+1,filename);
      perror("");exit(-1);
    }
    node[linecount].prior = atof(token);

    if ((token = strtok(NULL," \t\n")) == NULL) {
      node[linecount].mxcode = NULL;
    }
    else {
      node[linecount].mxcode = strdup(token);
    }

    i=strlen(node[linecount].name);
    j=strlen(UNKNAME);
    if (!strcmp(node[linecount].name, UNKNAME)) {
      node[linecount].isunk = 1;
    }
    else if ((i > j) && (node[linecount].name[i-j-1] == ',') && (!strcmp(node[linecount].name + i-j, UNKNAME))) {
      node[linecount].isunk = 1;
    }
    else {
      node[linecount].isunk = 0;
    }
    
    node[linecount].num_cnodes = 0;
    node[linecount].cnode_index = NULL;
    node[linecount].num_rseqs = 0;
    node[linecount].rseq_index = NULL;

    /* check that node[i].nid == i */
    if (node[linecount].nid != linecount) {
      fprintf(stderr,"ERROR (%s): node[i].nid != i, linecount %d file '%s'.\n",thisfunction,linecount+1,filename);
      fprintf(stderr,"note: taxonomy node indices must be consequtive integers starting from 0 (root).\n");
    }
    
    linecount++;
  }
  fclose(fp);
  
  /* count child nodes, here assuming node[i].nid == i */
  /* starting from i=1 since i=0 is root node and is not its cnode */
  for (i=1; i<(*num_nodes); i++) {
    node[ node[i].pid ].num_cnodes++;
  }

  for (i=0; i<(*num_nodes); i++) {
    if (node[i].num_cnodes > 0) {
      if ((node[i].cnode_index = (int *) malloc(node[i].num_cnodes * sizeof(int))) == NULL) {
	fprintf(stderr,"ERROR (%s): cannot malloc %d cnode int array for node %d/%d\n",thisfunction, node[i].num_cnodes,i+1,*num_nodes);
	perror(""); exit(-1);
      }
      /* zero since this is used as counter next */
      node[i].num_cnodes = 0;
    }
  }
  
  /* assign child node indices to parent node */
  /* starting from i=1 since i=0 is root node and is not its cnode */
  for (i=1; i<(*num_nodes); i++) {
    j=node[ node[i].pid ].num_cnodes;
    node[ node[i].pid ].cnode_index[j] = i;
    node[ node[i].pid ].num_cnodes++;
  }
  
  return(node);
}

int add_rseq2taxonomy(char *filename, TaxonomyNode *node) {
  FILE *fp;
  int nid,num,i, ind, linecount;
  char *thisfunction = "add_rseq2taxonomy";

  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot open '%s' for reading.\n",thisfunction,filename);
    perror(""); exit(-1);
  }
  /* line: nodeindex num seq1index seq2index ... seq[num]index */
  /* lines can be long, fgets is not used in order not to break any seqindex tokens => no need to detect/fix any split tokens */
  linecount=0;
  while (fscanf(fp,"%d %d",&nid,&num) == 2) {
    linecount++;
    if ((node[nid].rseq_index = (int *) malloc (num * sizeof(int))) == NULL) {
      fprintf(stderr,"ERROR (%s): cannot malloc node %d rseq_index with %d ints, ('linecount' %d) file '%s'.\n",thisfunction,nid,num,linecount,filename);
      perror(""); exit(-1);
    }    
    for (i=0; i<num; i++) {
      if (fscanf(fp,"%d",&ind) != 1) {
	fprintf(stderr,"ERROR (%s): cannot read node %d rseq_index %d/%d, ('linecount' %d) file '%s'.\n",thisfunction,nid,i+1,num,linecount,filename);
	perror(""); exit(-1);	
      }
      node[nid].rseq_index[i] = ind;
    }
    node[nid].num_rseqs=num;
  }
  
  fclose(fp);
  
  return(0);
}

int print_taxonomy(TaxonomyNode *node, int num_nodes) {
  int i;

  for (i=0; i<num_nodes; i++) {
    printf("%d %d %d %s %f unknode: %d num_cnodes: %d num_rseqs: %d\n",node[i].nid,node[i].pid,node[i].level,node[i].name,node[i].prior,node[i].isunk,node[i].num_cnodes,node[i].num_rseqs);
  }
  return(0);
}
