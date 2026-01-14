#ifndef ASCON128A_H
#define ASCON128A_H

#include <stdint.h>
#include <stdbool.h>

typedef uint64_t u64;
typedef uint8_t  u8;

#define ROUNDS_12 12
#define ROUNDS_8   8

// ================= API =================

// Encrypt
void ascon128a_encrypt(
  u8 *C, u8 *T,
  const u8 *P, int plen,
  const u8 *A, int alen,
  const u8 *N,
  const u8 *K
);

// Decrypt (returns true if tag valid)
bool ascon128a_decrypt(
  u8 *P,
  const u8 *C, int clen,
  const u8 *A, int alen,
  const u8 *N,
  const u8 *K,
  const u8 *T
);

#endif
