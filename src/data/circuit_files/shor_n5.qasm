OPENQASM 3.0;
include "stdgates.inc";
bit[5] c;
qubit[5] q;
x q[0];
h q[4];
h q[4];
c[0] = measure q[4];
reset q[4];
h q[4];
cx q[4], q[2];
cx q[4], q[0];
if (c == 1) {
  u1(pi/2) q[4];
}
h q[4];
c[1] = measure q[4];
reset q[4];
h q[4];
cswap q[4], q[1], q[0];
cswap q[4], q[2], q[1];
cswap q[4], q[3], q[2];
cx q[4], q[3];
cx q[4], q[2];
cx q[4], q[1];
cx q[4], q[0];
if (c == 3) {
  u1(3*pi/4) q[4];
}
if (c == 2) {
  u1(pi/2) q[4];
}
if (c == 1) {
  u1(pi/4) q[4];
}
h q[4];
c[2] = measure q[4];
