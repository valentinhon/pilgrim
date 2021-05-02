#ifndef _PILGRIM_LOG_FORMAT_H_
#define _PILGRIM_LOG_FORMAT_H_
#include <stdbool.h>
#include "mpi.h"
#include "uthash.h"

// Global variables
int g_mpi_rank;
int g_mpi_size;

// Store a list of lossless duration or interval
typedef struct TimingNode_t {
    double val;
    struct TimingNode_t *next;
} TimingNode;


typedef struct _Record {
    double tstart, tend;
    short func_id;              // 2 bytes function id
    int arg_count;
    int *arg_sizes;             // size of each argument
    void **args;                // Store all arguments in array
    int res;                    // result returned from the original function call
} Record;

/*
 * Entry of the Call Signature Table
 * key: call signature
 */
typedef struct RecordHash_t {
    void *key;                      // [func_id + arguments] as key
    int key_len;

    int rank;
    int terminal_id;                // terminal id used for sequitur compression
    double ext_tstart;              // last call's extrapolated tstart

    // statistics information
    double avg_duration;            // average duration
    double std_duration;            // standard deviation of the duration
    unsigned count;                 // count of this call signature

    // Lossless timing
    TimingNode *intervals;
    TimingNode *durations;

    UT_hash_handle hh;
} RecordHash;


typedef struct _LocalMetadata {
    int rank;
    double tstart;
    double tend;
    unsigned long records_count;
} LocalMetadata;


typedef struct _GlobalMetadata {
    double time_resolution;
    int ranks;
    int aggregated_timings;         // If aggreated (default) or non-aggregated timings are stored
} GlobalMetadata;


void logger_init();
void logger_exit();
void* compose_call_signature(Record *record, int *key_len);
void write_record(Record record);


bool is_recording();
void append_offset(MPI_Offset offset);


#endif
