##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AC_PREREQ(2.63)

m4_include([maint/version.m4])
AC_INIT([pilgrim],PILGRIM_VERSION_m4)

PILGRIM_VERSION=PILGRIM_VERSION_m4
AC_SUBST([PILGRIM_VERSION])

libpilgrim_so_version="libpilgrim_so_version_m4"
AC_SUBST([libpilgrim_so_version])

# Produce a numeric version assuming the following format:
# Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
# Example: 1.0.7rc1 has
#          MAJ = 1
#          MIN = 0
#          REV = 7
#          EXT = rc
#          EXT_NUMBER = 1
#
# Converting to numeric version will convert EXT to a format number:
#          ALPHA (a) = 0
#          BETA (b)  = 1
#          RC (rc)   = 2
#          PATCH (p) = 3
# Regular releases are treated as patch 0
#
# Numeric version will have 1 digit for MAJ, 2 digits for MIN,
# 2 digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER.
if test "${PILGRIM_VERSION}" = "unreleased" ; then
    PILGRIM_NUMVERSION=0
else
    changequote(<<,>>)
    V1=`expr $PILGRIM_VERSION : '\([0-9]*\)\.[0-9]*\.*[0-9]*[a-zA-Z]*[0-9]*'`
    V2=`expr $PILGRIM_VERSION : '[0-9]*\.\([0-9]*\)\.*[0-9]*[a-zA-Z]*[0-9]*'`
    V3=`expr $PILGRIM_VERSION : '[0-9]*\.[0-9]*\.*\([0-9]*\)[a-zA-Z]*[0-9]*'`
    V4=`expr $PILGRIM_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*\([a-zA-Z]*\)[0-9]*'`
    V5=`expr $PILGRIM_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*[a-zA-Z]*\([0-9]*\)'`
    changequote([,])

    if test "$V2" -le 9 ; then V2="0$V2" ; fi
    if test "$V3" = "" ; then V3="0"; fi
    if test "$V3" -le 9 ; then V3="0$V3" ; fi
    if test "$V4" = "a" ; then
	V4=0
    elif test "$V4" = "b" ; then
	V4=1
    elif test "$V4" = "rc" ; then
	V4=2
    elif test "$V4" = "" ; then
	V4=3
	V5=0
    elif test "$V4" = "p" ; then
	V4=3
    fi
    if test "$V5" -le 9 ; then V5="0$V5" ; fi

    PILGRIM_NUMVERSION=`expr $V1$V2$V3$V4$V5 + 0`
fi
AC_SUBST(PILGRIM_NUMVERSION)

AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_MACRO_DIR(m4)


dnl ----------------------------------------------------------------------------
dnl setup autotools
dnl ----------------------------------------------------------------------------
AM_INIT_AUTOMAKE([-Wall -Wno-portability-recursive -Werror foreign 1.12.3 subdir-objects])
AM_SILENT_RULES([yes])


dnl ----------------------------------------------------------------------------
dnl check the compiler and standard headers
dnl ----------------------------------------------------------------------------
dnl disable AC_PROG_CC's annoying default of adding -O2 to the CFLAGS
PAC_PUSH_FLAG([CFLAGS])
AC_PROG_CC([mpicc])
AC_PROG_CC_C99([mpicc])
PAC_POP_FLAG([CFLAGS])
AC_HEADER_STDC

# required pre-Automake-1.14
AM_PROG_CC_C_O

PAC_PUSH_FLAG([CFLAGS])
AM_PROG_AS
PAC_POP_FLAG([CFLAGS])

AM_PROG_AR

LT_INIT

AC_PROG_INSTALL


dnl ----------------------------------------------------------------------------
dnl configure options
dnl ----------------------------------------------------------------------------
# strict
PAC_ARG_STRICT

#PAC_APPEND_FLAG([-fno-common], [CFLAGS])
#AX_CHECK_COMPILE_FLAG([-fcommon],
#    [AX_APPEND_FLAG([-fcommon])],
#    [AC_MSG_WARN([-fcommon not supported, you may have to set CFLAGS to avoid multiple definitions error.])
#])

AC_ARG_WITH(zfp,
	AS_HELP_STRING([--with-zfp=DIR], [Specify the install location of ZFP]),,
	with_zfp=)
ZFP_DIR="$with_zfp"
AC_SUBST([ZFP_DIR])
AM_CONDITIONAL([WITH_ZFP],[test "$with_zfp" != ""])
if test "${with_zfp}" != "" ; then
    PAC_APPEND_FLAG([-DWITH_ZFP], [CFLAGS])
fi

AC_ARG_WITH(sz,
	AS_HELP_STRING([--with-sz=DIR], [Specify the install location of SZ]),,
	with_sz=)
SZ_DIR="$with_sz"
AC_SUBST([SZ_DIR])
AM_CONDITIONAL([WITH_SZ],[test "$with_sz" != ""])
if test "${with_sz}" != "" ; then
    PAC_APPEND_FLAG([-DWITH_SZ], [CFLAGS])
fi

# --enable-debug
AC_ARG_ENABLE([g],AS_HELP_STRING([--enable-g],[alias for --enable-debug]),,[enable_g=no])
AC_ARG_ENABLE([debug],AS_HELP_STRING([--enable-debug],[adds -g to CFLAGS]),,[enable_debug=${enable_g}])
if test "${enable_debug}" != "no" ; then
    PAC_APPEND_FLAG([-g],[CFLAGS])
fi

AC_ARG_ENABLE([pointers], AS_HELP_STRING([--enable-pointers], [Enabel intercepting and storing pointers [default=no]]),,[enable_pointers=no])
if test "${enable_pointers}" == "yes" ; then
    PAC_APPEND_FLAG([-DMEMORY_POINTERS], [CFLAGS])
fi

AC_ARG_ENABLE([cuda], AS_HELP_STRING([--enable-cuda], [Enabel intercepting cuda memory calls and storing device locations [default=no]]),,[enable_cuda=no])
if test "${enable_cuda}" == "yes" ; then
    PAC_APPEND_FLAG([-DCUDA_POINTERS], [CFLAGS])
    PAC_APPEND_FLAG([-lcuda -lcudart], [LDFLAGS])
fi

AC_ARG_ENABLE([tid], AS_HELP_STRING([--enable-tid], [Storing the thread id of each call [default=no]]),,[enable_tid=no])
if test "${enable_tid}" == "yes" ; then
    PAC_APPEND_FLAG([-DENABLE_THREAD_ID], [CFLAGS])
fi




dnl ----------------------------------------------------------------------------
dnl check the environment and the availability of functions
dnl ----------------------------------------------------------------------------
PAC_C_GNU_ATTRIBUTE

# look for pthreads
AC_CHECK_HEADERS(pthread.h)
AC_CHECK_LIB([pthread],[pthread_key_create],have_pthreads=yes)
if test "$have_pthreads" = "yes" ; then
    PAC_PREPEND_FLAG([-lpthread],[LIBS])
else
    AC_ERROR([pthreads not found on the system])
fi

# look for zstd
AC_CHECK_HEADERS(zstd.h)
AC_CHECK_LIB([zstd],[ZSTD_compressBound],have_zstd=yes)
if test "$have_zstd" = "yes" ; then
    PAC_PREPEND_FLAG([-lzstd],[LIBS])
else
    if test "${with_sz}" == "" ; then
        AC_ERROR([zstd not found on the system, please install libzstd-dev])
    fi
fi


dnl ----------------------------------------------------------------------------
dnl Final setup
dnl ----------------------------------------------------------------------------


dnl ----------------------------------------------------------------------------
dnl config headers
dnl ----------------------------------------------------------------------------
AC_CONFIG_HEADERS([include/pilgrim_config.h])
AH_TOP([
/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef PILGRIM_CONFIG_H_INCLUDED
#define PILGRIM_CONFIG_H_INCLUDED
])
AH_BOTTOM([
#endif /* PILGRIM_CONFIG_H_INCLUDED */
])

AC_CONFIG_FILES([Makefile                       \
                 test/Makefile                  \
                 test/tracing/Makefile          \
                 test/mpi/misc/Makefile         \
                 test/mpi/pt2pt/Makefile        \
                 test/mpi/coll/Makefile         \
                 test/mpi/util/Makefile         \
                ])
AC_OUTPUT
