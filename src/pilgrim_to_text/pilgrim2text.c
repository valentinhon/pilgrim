#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "pilgrim.h"
#include "pilgrim_reader.h"

int main(int argc, char** argv) {

    char* directory = argv[1];
    char cfg_path[256];
    char cst_path[256];
    sprintf(cfg_path, "%s/grammars.dat", directory);
    sprintf(cst_path, "%s/funcs.dat", directory);

    // 1. Read CST
    int num_sigs;
    CallSignature *cst = read_cst(cst_path, &num_sigs);

    // 2. Read CFG
    int nprocs = atoi(argv[2]);
    int num_symbols[nprocs];
    int** decoded_symbols = read_cfg(cfg_path, nprocs, num_symbols);

    // 3. Write to text files
    char textfile_dir[256], textfile_path[256];
    sprintf(textfile_dir, "%s/_text", directory);
    mkdir(textfile_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for(int rank = 0; rank < nprocs; rank++) {
        sprintf(textfile_path, "%s/%d.txt", textfile_dir, rank);
        FILE* f = fopen(textfile_path, "w");
        for(int i = 0; i < num_symbols[rank]; i+=2) {

            int sym = decoded_symbols[rank][i];
            int exp = decoded_symbols[rank][i+1];
            CallSignature cs = cst[sym];
            for(int j = 0; j < exp; j++) {
                fprintf(f, "%s\n", func_names[cs.func_id]);
            }
        }

        fclose(f);
        free(decoded_symbols[rank]);
    }
    free(decoded_symbols);

    return 0;
}
