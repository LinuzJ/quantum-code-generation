OPENQASM 3.0;
include "stdgates.inc";
bit[4] c;
qubit[4] q;
x q[0];
x q[1];
rz(pi/4) q[1];
rz(-pi/4) q[2];
cx q[1], q[2];
h q[1];
cx q[2], q[1];
rz(1.5632210989912343) q[1];
cx q[2], q[1];
rz(-1.5632210989912343) q[1];
h q[1];
cx q[1], q[2];
rz(-pi/4) q[1];
rz(pi/4) q[2];
rz(0) q[2];
rz(pi/4) q[0];
rz(-pi/4) q[1];
cx q[0], q[1];
h q[0];
cx q[1], q[0];
rz(-0.7892146885910722) q[0];
cx q[1], q[0];
rz(0.7892146885910722) q[0];
h q[0];
cx q[0], q[1];
rz(-pi/4) q[0];
rz(pi/4) q[1];
rz(0) q[1];
rz(pi/4) q[2];
rz(-pi/4) q[3];
cx q[2], q[3];
h q[2];
cx q[3], q[2];
rz(-0.7816390259909868) q[2];
cx q[3], q[2];
rz(0.7816390259909868) q[2];
h q[2];
cx q[2], q[3];
rz(-pi/4) q[2];
rz(pi/4) q[3];
rz(0) q[3];
rz(pi/4) q[1];
rz(-pi/4) q[2];
cx q[1], q[2];
h q[1];
cx q[2], q[1];
rz(0.0075754452018738224) q[1];
cx q[2], q[1];
rz(-0.0075754452018738224) q[1];
h q[1];
cx q[1], q[2];
rz(-pi/4) q[1];
rz(pi/4) q[2];
rz(0) q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
