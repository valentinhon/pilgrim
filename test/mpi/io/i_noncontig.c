/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpitest.h"

/* tests noncontiguous reads/writes using nonblocking I/O */

/*
static char MTEST_Descrip[] = "Test nonblocking I/O";
*/

#define SIZE 5000

#define VERBOSE 0

#define HANDLE_ERROR(err) \
    if (err != MPI_SUCCESS) { \
        char msg[MPI_MAX_ERROR_STRING]; \
        int resultlen; \
        MPI_Error_string(err, msg, &resultlen); \
        fprintf(stderr, "%s line %d: %s\n", __FILE__, __LINE__, msg); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    }

int main(int argc, char **argv)
{
    int *buf, i, mynod, nprocs, len, blocklength;
    int err, errs = 0;
    MPI_Aint displacement;
    MPI_File fh;
    MPI_Status status;
    char *filename;
    MPI_Datatype typevec, typevec2, newtype;
    MPI_Request req;

    MTest_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynod);

    if (nprocs != 2) {
        fprintf(stderr, "Run this program on two processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

/* process 0 takes the file name as a command-line argument and
   broadcasts it to other processes */
    if (!mynod) {
        i = 1;
        while ((i < argc) && strcmp("-fname", *argv)) {
            i++;
            argv++;
        }
        if (i >= argc) {
            len = 8;
            filename = (char *) malloc(len + 10);
            strcpy(filename, "testfile");
            /*
             * fprintf(stderr, "\n*#  Usage: i_noncontig -fname filename\n\n");
             * MPI_Abort(MPI_COMM_WORLD, 1);
             */
        } else {
            argv++;
            len = (int) strlen(*argv);
            filename = (char *) malloc(len + 1);
            strcpy(filename, *argv);
        }
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(filename, len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        filename = (char *) malloc(len + 1);
        MPI_Bcast(filename, len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    buf = (int *) malloc(SIZE * sizeof(int));

    MPI_Type_vector(SIZE / 2, 1, 2, MPI_INT, &typevec);

    blocklength = 1;
    displacement = mynod * sizeof(int);

    MPI_Type_create_struct(1, &blocklength, &displacement, &typevec, &typevec2);
    MPI_Type_create_resized(typevec2, 0, SIZE * sizeof(int), &newtype);
    MPI_Type_commit(&newtype);
    MPI_Type_free(&typevec);
    MPI_Type_free(&typevec2);

    if (!mynod) {
#if VERBOSE
        fprintf(stderr,
                "\ntesting noncontiguous in memory, noncontiguous in file using nonblocking I/O\n");
#endif
        MPI_File_delete(filename, MPI_INFO_NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    err =
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                      &fh);
    HANDLE_ERROR(err);

    err = MPI_File_set_view(fh, 0, MPI_INT, newtype, (char *) "native", MPI_INFO_NULL);
    HANDLE_ERROR(err);

    for (i = 0; i < SIZE; i++)
        buf[i] = i + mynod * SIZE;
    err = MPI_File_iwrite(fh, buf, 1, newtype, &req);
    HANDLE_ERROR(err);
    err = MPI_Wait(&req, &status);
    HANDLE_ERROR(err);

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++)
        buf[i] = -1;

    err = MPI_File_iread_at(fh, 0, buf, 1, newtype, &req);
    HANDLE_ERROR(err);
    err = MPI_Wait(&req, &status);
    HANDLE_ERROR(err);

    for (i = 0; i < SIZE; i++) {
        if (!mynod) {
            if ((i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
            if (!(i % 2) && (buf[i] != i)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n", mynod, i, buf[i], i);
            }
        } else {
            if ((i % 2) && (buf[i] != i + mynod * SIZE)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n",
                        mynod, i, buf[i], i + mynod * SIZE);
            }
            if (!(i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
        }
    }

    err = MPI_File_close(&fh);
    HANDLE_ERROR(err);

    MPI_Barrier(MPI_COMM_WORLD);

    if (!mynod) {
#if VERBOSE
        fprintf(stderr,
                "\ntesting noncontiguous in memory, contiguous in file using nonblocking I/O\n");
#endif
        MPI_File_delete(filename, MPI_INFO_NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    err =
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                      &fh);
    HANDLE_ERROR(err);

    for (i = 0; i < SIZE; i++)
        buf[i] = i + mynod * SIZE;
    err = MPI_File_iwrite_at(fh, mynod * (SIZE / 2) * sizeof(int), buf, 1, newtype, &req);
    HANDLE_ERROR(err);
    err = MPI_Wait(&req, &status);
    HANDLE_ERROR(err);

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++)
        buf[i] = -1;

    err = MPI_File_iread_at(fh, mynod * (SIZE / 2) * sizeof(int), buf, 1, newtype, &req);
    HANDLE_ERROR(err);
    err = MPI_Wait(&req, &status);
    HANDLE_ERROR(err);

    for (i = 0; i < SIZE; i++) {
        if (!mynod) {
            if ((i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
            if (!(i % 2) && (buf[i] != i)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n", mynod, i, buf[i], i);
            }
        } else {
            if ((i % 2) && (buf[i] != i + mynod * SIZE)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n",
                        mynod, i, buf[i], i + mynod * SIZE);
            }
            if (!(i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
        }
    }

    err = MPI_File_close(&fh);
    HANDLE_ERROR(err);

    MPI_Barrier(MPI_COMM_WORLD);

    if (!mynod) {
#if VERBOSE
        fprintf(stderr,
                "\ntesting contiguous in memory, noncontiguous in file using nonblocking I/O\n");
#endif
        MPI_File_delete(filename, MPI_INFO_NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    err =
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                      &fh);
    HANDLE_ERROR(err);

    err = MPI_File_set_view(fh, 0, MPI_INT, newtype, (char *) "native", MPI_INFO_NULL);
    HANDLE_ERROR(err);

    for (i = 0; i < SIZE; i++)
        buf[i] = i + mynod * SIZE;
    err = MPI_File_iwrite(fh, buf, SIZE, MPI_INT, &req);
    HANDLE_ERROR(err);
    err = MPI_Wait(&req, &status);
    HANDLE_ERROR(err);

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++)
        buf[i] = -1;

    err = MPI_File_iread_at(fh, 0, buf, SIZE, MPI_INT, &req);
    HANDLE_ERROR(err);
    err = MPI_Wait(&req, &status);
    HANDLE_ERROR(err);

    for (i = 0; i < SIZE; i++) {
        if (!mynod) {
            if (buf[i] != i) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n", mynod, i, buf[i], i);
            }
        } else {
            if (buf[i] != i + mynod * SIZE) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n",
                        mynod, i, buf[i], i + mynod * SIZE);
            }
        }
    }

    err = MPI_File_close(&fh);
    HANDLE_ERROR(err);

    MPI_Type_free(&newtype);
    free(buf);
    free(filename);
    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
