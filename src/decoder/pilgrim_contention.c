//
// Created by khatharsis on 21/05/24.
//

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include "pilgrim.h"
#include "pilgrim_timings.h"
#include "pilgrim_reader.h"


void process_record(int rank, CallSignature* cs,int size, unsigned int matrix[size][size]) {

    /* Second argument of MPI_send or MPI_Isend: number of elements in the buffer
    * Store it in mpi_count to compute the size of the communication
    */
    unsigned int* mpi_count = (unsigned int*)cs->args[1];

    /* Third argument of MPI_send or MPI_Isend: MPI_data_type
    * Store it in mpi_data_size to compute the size of the communication
    */
    MPI_Datatype* mpi_data_size =(MPI_Datatype*) cs->args[2];

    /* Compute size of the communication in bytes */
    unsigned int comm_size = *mpi_count * sizeof(*mpi_data_size);

    /* Get the destination rank */
    int* mpi_destination =  (unsigned int*)cs->args[3];
    /* As it is relative, compute the destination with the value of sender rank and relative destination rank */
    int destination = rank - *mpi_destination;
    /* Be careful to stay modulo rank */
    if (destination<0)
    {
        destination = size-1-destination;
    }
    /* Security check */
    assert (destination>=0);

    /* Increment matrix count for rank rank and destination mpi_destination */
    matrix[rank][destination] += comm_size;
}





void write_matrix(char* filename, int size, unsigned int matrix[size][size]){

    /* Open the file to store the matrix*/
    FILE* file = fopen(filename,"w+");
    if (file==NULL)
    {
        fprintf(stderr,"[PILGRIM COMMUNICATION MATRIX] Error in opening the output file to store the matrix : %s\n",filename);
        exit(-1);
    }

    /* Write the first line containing all the ranks id */
    for (int i=0; i<size; i++){
        fprintf(file,"%d",i+1);
        if (i<size-1)
            fprintf(file,",");
    }
    fprintf(file,"\n");

    /* Write the content of the matrix rank by rank */
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(file,"%u",matrix[i][j]);
            if (j<size-1)
                fprintf(file,",");
        }
        fprintf(file,"\n");
    }

    /* Close file descriptor before leaving */
    fclose(file);
}



void usage(const char* prog_name) {
    printf("Usage: %s -t trace_file [OPTION] \n", prog_name);
    printf("\t-t FILE   --- Path to the Pallas trace file\n");
    printf("\t[OPTION] -n OUTPUT_FILE   --- Name of the output file. Extension should be .csi. By default, the name of the file with be pilgrim_contention.sci\n");
    printf("\t[OPTION] -? -h   --- Display this help and exit\n");
}

int main(int argc, char** argv) {
    /* Print usage if asked with -h or -? */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    /* If too few arguments or too much arguments, exit
      * Case of help options already treated above
      * Either one argument (path to trace) or two (path to trace and matrix file name)
     */
    if (argc<2 || argc >3)
    {
        usage(argv[0]);
        return EXIT_SUCCESS;
    }

    /* Default name for the pilgrim contention file */
    char*  output_file="pilgrim_contention.csi";
    /* If there is a given file name to store the matrix */
    if (argc==3)
        output_file=argv[2];

    /* Starting processing of trace */
    printf("[pilgrim-contention] Starting contention calculation...\n");
    struct timespec start, finish;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    // 0. Read metadata
    GlobalMetadata *gm = read_metadata(argv[1]);
    double *tstarts = read_tstarts(gm);
    double *tends   = read_tends(gm);
    double *p_tstarts = tstarts;
    double *p_tends   = tends;

    // 1. Read CST and CFG
    CST *cst = read_cst(gm);
    CFG* cfg = read_cfg(gm);
    void* matrix;

    for(int rank = 0; rank < gm->ranks; rank++) {
        int ugi = cfg->grammar_ids[rank];

        int total_calls = 0;
        for(int i=0; i<cfg->num_symbols[ugi]; i+=2)
            total_calls += cfg->unique_grammars[ugi][i+1];
        printf("Rank %d, total number of calls: %d\n", rank, total_calls);
        const char* textfile_path = "test.txt";
        const char* textfile_dir="./";
        sprintf(textfile_path, "%s/%d.txt", textfile_dir, rank);
        FILE* f = fopen(textfile_path, "w");
        for(int i = 0; i < cfg->num_symbols[ugi]; i+=2) {

            int sym = cfg->unique_grammars[ugi][i];
            int exp = cfg->unique_grammars[ugi][i+1];
            CallSignature *cs = &(cst->cs_list[sym]);
            for(int j = 0; j < exp; j++) {
                if(strcmp(gm->timing_mode, TIMING_MODE_LOSSLESS)==0)
                    fprintf(f, "%f %f ", p_tstarts[i/2], p_tends[i/2]);
                fprintf(f, "%s()\n", func_names[cs->func_id]);
            }
        }
        fclose(f);

        p_tstarts += total_calls;
        p_tends   += total_calls;
    }

    free_metadata(gm);
    free_cst(cst);
    free_cfg(cfg);
    free(tstarts);
    free(tends);


    // 3. For each rank, parse events and look for MPI_send
    for(int rank = 0; rank < gm->ranks; rank++) {
        int ugi = cfg->grammar_ids[rank];

        for(int i = 0; i < cfg->num_symbols[ugi]; i+=2) {

            int sym = cfg->unique_grammars[ugi][i];
            CallSignature *cs = &(cst->cs_list[sym]);

            if (cs->func_id ==ID_MPI_Send || cs->func_id== ID_MPI_Isend)
            {
                // Fill the matrix by reading parameters of the MPI_send in CallSignature* cs
                process_record(rank, cs, gm->ranks, matrix);
            }
        }
    }

    /// Flush matrix in output file
    char textfile_path[512];
    // Merge path to the pilgrim trace and matrix file name to store the file at the same place as the trace
    sprintf(textfile_path, "%s%s", argv[1],output_file);
    write_matrix(textfile_path, gm->ranks, matrix);

    // 4. Free data
    free_metadata(gm);
    free_cst(cst);
    free_cfg(cfg);

    /* Print program duration */
    //clock_gettime(CLOCK_MONOTONIC, &finish);
    //double time_elapsed = finish.tv_nsec-start.tv_nsec + (finish.tv_sec - start.tv_sec) * 1000000000;
    //printf("\n[pilgrim-comm-matrix] *** Elapsed time: %lf nanoseconds. ***\n",time_elapsed);

    return 0;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
