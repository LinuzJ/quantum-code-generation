OPENQASM 3.0;
include "stdgates.inc";
qubit[4] reg;
h reg[3];
h reg[2];
h reg[0];
h reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
h reg[3];
h reg[2];
h reg[0];
h reg[0];
y reg[3];
y reg[2];
y reg[0];
y reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
y reg[3];
y reg[2];
y reg[0];
y reg[0];
h reg[3];
y reg[2];
h reg[0];
y reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
h reg[3];
y reg[2];
h reg[0];
y reg[0];
y reg[3];
h reg[2];
y reg[0];
h reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
y reg[3];
h reg[2];
y reg[0];
h reg[0];
y reg[3];
y reg[2];
h reg[0];
h reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
y reg[3];
y reg[2];
h reg[0];
h reg[0];
h reg[3];
h reg[2];
y reg[0];
y reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
h reg[3];
h reg[2];
y reg[0];
y reg[0];
y reg[3];
h reg[2];
h reg[0];
y reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
y reg[3];
h reg[2];
h reg[0];
y reg[0];
h reg[3];
y reg[2];
y reg[0];
h reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(2.151746) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
h reg[3];
y reg[2];
y reg[0];
h reg[0];
h reg[1];
h reg[0];
cx reg[1], reg[0];
rz(1.995482) reg[0];
cx reg[1], reg[0];
h reg[1];
h reg[0];
y reg[1];
y reg[0];
cx reg[1], reg[0];
rz(1.995482) reg[0];
cx reg[1], reg[0];
y reg[1];
y reg[0];
h reg[2];
h reg[0];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(4.332582) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
h reg[2];
h reg[0];
y reg[2];
y reg[0];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(4.332582) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
y reg[2];
y reg[0];
h reg[2];
h reg[1];
cx reg[2], reg[1];
rz(0.461922) reg[1];
cx reg[2], reg[1];
h reg[2];
h reg[1];
y reg[2];
y reg[1];
cx reg[2], reg[1];
rz(0.461922) reg[1];
cx reg[2], reg[1];
y reg[2];
y reg[1];
h reg[3];
h reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(1.086976) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
h reg[3];
h reg[0];
y reg[3];
y reg[0];
cx reg[3], reg[2];
cx reg[2], reg[1];
cx reg[1], reg[0];
rz(1.086976) reg[0];
cx reg[1], reg[0];
cx reg[2], reg[1];
cx reg[3], reg[2];
y reg[3];
y reg[0];
h reg[3];
h reg[1];
cx reg[3], reg[2];
cx reg[2], reg[1];
rz(2.258394) reg[1];
cx reg[2], reg[1];
cx reg[3], reg[2];
h reg[3];
h reg[1];
y reg[3];
y reg[1];
cx reg[3], reg[2];
cx reg[2], reg[1];
rz(2.258394) reg[1];
cx reg[2], reg[1];
cx reg[3], reg[2];
y reg[3];
y reg[1];
h reg[3];
h reg[2];
cx reg[3], reg[2];
rz(1.228531) reg[2];
cx reg[3], reg[2];
h reg[3];
h reg[2];
y reg[3];
y reg[2];
cx reg[3], reg[2];
rz(1.228531) reg[2];
cx reg[3], reg[2];
y reg[3];
y reg[2];
