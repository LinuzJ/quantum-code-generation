OPENQASM 3.0;
include "stdgates.inc";
bit[5] c;
qubit[5] q;
h q[0];
h q[1];
h q[3];
h q[4];
cx q[3], q[2];
cx q[0], q[2];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
