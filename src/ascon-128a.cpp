#include "ascon-128a.h"
#include <string.h>

// ================= Rotation =================
static inline u64 ROTR(u64 x, int n) {
  return (x >> n) | (x << (64 - n));
}

// ================= Load / Store =================
static u64 load64(const u8 *b) {
  u64 x = 0;
  for (int i = 0; i < 8; i++) x = (x << 8) | b[i];
  return x;
}

static void store64(u8 *b, u64 x) {
  for (int i = 7; i >= 0; i--) {
    b[i] = x & 0xFF;
    x >>= 8;
  }
}

// ================= ASCON Permutation =================
static void ascon_permutation(u64 s[5], int rounds) {
  static const u64 RC[12] = {
    0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,
    0x96,0x87,0x78,0x69,0x5a,0x4b
  };

  for (int r = 12 - rounds; r < 12; r++) {
    s[2] ^= RC[r];

    s[0] ^= s[4]; s[4] ^= s[3]; s[2] ^= s[1];

    u64 t0 = ~s[0] & s[1];
    u64 t1 = ~s[1] & s[2];
    u64 t2 = ~s[2] & s[3];
    u64 t3 = ~s[3] & s[4];
    u64 t4 = ~s[4] & s[0];

    s[0] ^= t1; s[1] ^= t2; s[2] ^= t3;
    s[3] ^= t4; s[4] ^= t0;

    s[1] ^= s[0]; s[0] ^= s[4];
    s[3] ^= s[2]; s[2] = ~s[2];

    s[0] ^= ROTR(s[0],19) ^ ROTR(s[0],28);
    s[1] ^= ROTR(s[1],61) ^ ROTR(s[1],39);
    s[2] ^= ROTR(s[2], 1) ^ ROTR(s[2], 6);
    s[3] ^= ROTR(s[3],10) ^ ROTR(s[3],17);
    s[4] ^= ROTR(s[4], 7) ^ ROTR(s[4],41);
  }
}

// ================= ASCON-128a ENCRYPT =================
void ascon128a_encrypt(
  u8 *C, u8 *T,
  const u8 *P, int plen,
  const u8 *A, int alen,
  const u8 *N,
  const u8 *K
) {
  u64 S[5];

  S[0] = 0x00001000808c0002ULL;
  S[1] = load64(K);
  S[2] = load64(K + 8);
  S[3] = load64(N);
  S[4] = load64(N + 8);

  ascon_permutation(S, ROUNDS_12);
  S[3] ^= load64(K);
  S[4] ^= load64(K + 8);

  if (alen > 0) {
    while (alen >= 8) {
      S[0] ^= load64(A);
      ascon_permutation(S, ROUNDS_8);
      A += 8; alen -= 8;
    }
    u8 lastA[8] = {0};
    memcpy(lastA, A, alen);
    lastA[alen] = 0x80;
    S[0] ^= load64(lastA);
    ascon_permutation(S, ROUNDS_8);
  }

  S[4] ^= 1;

  while (plen >= 8) {
    S[0] ^= load64(P);
    store64(C, S[0]);
    ascon_permutation(S, ROUNDS_8);
    P += 8; C += 8; plen -= 8;
  }

  u8 lastP[8] = {0};
  memcpy(lastP, P, plen);
  lastP[plen] = 0x80;
  S[0] ^= load64(lastP);

  if (plen > 0) {
    u8 tmp[8];
    store64(tmp, S[0]);
    memcpy(C, tmp, plen);
  }

  S[1] ^= load64(K);
  S[2] ^= load64(K + 8);
  ascon_permutation(S, ROUNDS_12);
  S[3] ^= load64(K);
  S[4] ^= load64(K + 8);

  store64(T,     S[3]);
  store64(T + 8, S[4]);
}

// ================= ASCON-128a DECRYPT =================
bool ascon128a_decrypt(
  u8 *P,
  const u8 *C, int clen,
  const u8 *A, int alen,
  const u8 *N,
  const u8 *K,
  const u8 *T
) {
  u64 S[5];

  S[0] = 0x00001000808c0002ULL;
  S[1] = load64(K);
  S[2] = load64(K + 8);
  S[3] = load64(N);
  S[4] = load64(N + 8);

  ascon_permutation(S, ROUNDS_12);
  S[3] ^= load64(K);
  S[4] ^= load64(K + 8);

  if (alen > 0) {
    while (alen >= 8) {
      S[0] ^= load64(A);
      ascon_permutation(S, ROUNDS_8);
      A += 8; alen -= 8;
    }
    u8 lastA[8] = {0};
    memcpy(lastA, A, alen);
    lastA[alen] = 0x80;
    S[0] ^= load64(lastA);
    ascon_permutation(S, ROUNDS_8);
  }

  S[4] ^= 1;

  while (clen >= 8) {
    u64 c = load64(C);
    u64 p = S[0] ^ c;
    store64(P, p);
    S[0] = c;
    ascon_permutation(S, ROUNDS_8);
    C += 8; P += 8; clen -= 8;
  }

  u8 tmp[8];
  store64(tmp, S[0]);
  for (int i = 0; i < clen; i++) {
    P[i] = tmp[i] ^ C[i];
    tmp[i] = C[i];
  }
  tmp[clen] ^= 0x80;
  S[0] = load64(tmp);

  S[1] ^= load64(K);
  S[2] ^= load64(K + 8);
  ascon_permutation(S, ROUNDS_12);
  S[3] ^= load64(K);
  S[4] ^= load64(K + 8);

  u8 Tcalc[16];
  store64(Tcalc, S[3]);
  store64(Tcalc + 8, S[4]);

  return memcmp(Tcalc, T, 16) == 0;
}
