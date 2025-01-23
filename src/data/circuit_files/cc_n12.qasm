OPENQASM 3.0;
include "stdgates.inc";
bit[12] cr;
qubit[12] qr;
h qr[0];
h qr[1];
h qr[2];
h qr[3];
h qr[4];
h qr[5];
h qr[6];
h qr[7];
h qr[8];
h qr[9];
h qr[10];
cx qr[0], qr[11];
cx qr[1], qr[11];
cx qr[2], qr[11];
cx qr[3], qr[11];
cx qr[4], qr[11];
cx qr[5], qr[11];
cx qr[6], qr[11];
cx qr[7], qr[11];
cx qr[8], qr[11];
cx qr[9], qr[11];
cx qr[10], qr[11];
cr[11] = measure qr[11];
if (cr == 0) {
  x qr[11];
}
if (cr == 0) {
  h qr[11];
}
if (cr == 2048) {
  h qr[0];
}
if (cr == 2048) {
  h qr[1];
}
if (cr == 2048) {
  h qr[2];
}
if (cr == 2048) {
  h qr[3];
}
if (cr == 2048) {
  h qr[4];
}
if (cr == 2048) {
  h qr[5];
}
if (cr == 2048) {
  h qr[6];
}
if (cr == 2048) {
  h qr[7];
}
if (cr == 2048) {
  h qr[8];
}
if (cr == 2048) {
  h qr[9];
}
if (cr == 2048) {
  h qr[10];
}
barrier qr[0], qr[1], qr[2], qr[3], qr[4], qr[5], qr[6], qr[7], qr[8], qr[9], qr[10], qr[11];
if (cr == 0) {
  cx qr[6], qr[11];
}
barrier qr[0], qr[1], qr[2], qr[3], qr[4], qr[5], qr[6], qr[7], qr[8], qr[9], qr[10], qr[11];
if (cr == 0) {
  h qr[0];
}
if (cr == 0) {
  h qr[1];
}
if (cr == 0) {
  h qr[2];
}
if (cr == 0) {
  h qr[3];
}
if (cr == 0) {
  h qr[4];
}
if (cr == 0) {
  h qr[5];
}
if (cr == 0) {
  h qr[6];
}
if (cr == 0) {
  h qr[7];
}
if (cr == 0) {
  h qr[8];
}
if (cr == 0) {
  h qr[9];
}
if (cr == 0) {
  h qr[10];
}
cr[0] = measure qr[0];
cr[1] = measure qr[1];
cr[2] = measure qr[2];
cr[3] = measure qr[3];
cr[4] = measure qr[4];
cr[5] = measure qr[5];
cr[6] = measure qr[6];
cr[7] = measure qr[7];
cr[8] = measure qr[8];
cr[9] = measure qr[9];
cr[10] = measure qr[10];
