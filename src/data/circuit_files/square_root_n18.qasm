OPENQASM 3.0;
include "stdgates.inc";
bit[13] c;
qubit[18] q;
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
z q[17];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[1], q[0], q[13];
ccx q[13], q[2], q[14];
ccx q[14], q[3], q[15];
ccx q[15], q[4], q[16];
ccx q[16], q[5], q[17];
z q[17];
ccx q[16], q[5], q[17];
ccx q[15], q[4], q[16];
ccx q[14], q[3], q[15];
ccx q[13], q[2], q[14];
ccx q[1], q[0], q[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
z q[17];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[1], q[0], q[13];
ccx q[13], q[2], q[14];
ccx q[14], q[3], q[15];
ccx q[15], q[4], q[16];
ccx q[16], q[5], q[17];
z q[17];
ccx q[16], q[5], q[17];
ccx q[15], q[4], q[16];
ccx q[14], q[3], q[15];
ccx q[13], q[2], q[14];
ccx q[1], q[0], q[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
z q[17];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[1], q[0], q[13];
ccx q[13], q[2], q[14];
ccx q[14], q[3], q[15];
ccx q[15], q[4], q[16];
ccx q[16], q[5], q[17];
z q[17];
ccx q[16], q[5], q[17];
ccx q[15], q[4], q[16];
ccx q[14], q[3], q[15];
ccx q[13], q[2], q[14];
ccx q[1], q[0], q[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
z q[17];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[1], q[0], q[13];
ccx q[13], q[2], q[14];
ccx q[14], q[3], q[15];
ccx q[15], q[4], q[16];
ccx q[16], q[5], q[17];
z q[17];
ccx q[16], q[5], q[17];
ccx q[15], q[4], q[16];
ccx q[14], q[3], q[15];
ccx q[13], q[2], q[14];
ccx q[1], q[0], q[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
z q[17];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[1], q[0], q[13];
ccx q[13], q[2], q[14];
ccx q[14], q[3], q[15];
ccx q[15], q[4], q[16];
ccx q[16], q[5], q[17];
z q[17];
ccx q[16], q[5], q[17];
ccx q[15], q[4], q[16];
ccx q[14], q[3], q[15];
ccx q[13], q[2], q[14];
ccx q[1], q[0], q[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
z q[17];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[1], q[0], q[13];
ccx q[13], q[2], q[14];
ccx q[14], q[3], q[15];
ccx q[15], q[4], q[16];
ccx q[16], q[5], q[17];
z q[17];
ccx q[16], q[5], q[17];
ccx q[15], q[4], q[16];
ccx q[14], q[3], q[15];
ccx q[13], q[2], q[14];
ccx q[1], q[0], q[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0], q[6];
cx q[1], q[8];
cx q[2], q[10];
cx q[3], q[6];
cx q[3], q[7];
cx q[4], q[8];
cx q[4], q[9];
cx q[5], q[10];
cx q[5], q[11];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
ccx q[7], q[6], q[13];
ccx q[13], q[8], q[14];
ccx q[14], q[9], q[15];
ccx q[15], q[10], q[16];
ccx q[16], q[11], q[17];
cx q[17], q[12];
ccx q[16], q[11], q[17];
ccx q[15], q[10], q[16];
ccx q[14], q[9], q[15];
ccx q[13], q[8], q[14];
ccx q[7], q[6], q[13];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
c[12] = measure q[12];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
c[6] = measure q[6];
c[7] = measure q[7];
c[8] = measure q[8];
c[9] = measure q[9];
c[10] = measure q[10];
c[11] = measure q[11];
