//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

#include <stdint.h>

//-----------------------------------------------------------------------------

uint32_t MurmurHash3_x86_32  ( const void * key, int len, uint32_t seed );

//-----------------------------------------------------------------------------

#endif // _MURMURHASH3_H_
