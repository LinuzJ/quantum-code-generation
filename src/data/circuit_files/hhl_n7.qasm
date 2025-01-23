OPENQASM 3.0;
include "stdgates.inc";
bit[7] meas;
qubit[1] q0;
qubit[5] q1;
qubit[1] q2;
rz(-pi/4) q0[0];
ry(pi) q0[0];
rz(pi/4) q0[0];
rx(3.8098602) q1[0];
ry(-pi/2) q1[0];
cx q1[0], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[0], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(-1.7108829) q1[0];
rx(3.8098602) q1[1];
ry(-pi/2) q1[1];
cx q1[1], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[1], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[1];
cx q1[1], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[1], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(-3.4217658) q1[1];
rx(3.8098602) q1[2];
ry(-pi/2) q1[2];
cx q1[2], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[2];
cx q1[2], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[2];
cx q1[2], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[2];
cx q1[2], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(-0.56034639) q1[2];
rx(3.8098602) q1[3];
ry(-pi/2) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[3];
cx q1[3], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(-1.1206928) q1[3];
rx(3.8098602) q1[4];
ry(-pi/2) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(0.6682675) q0[0];
rz(0.6682675) q1[4];
cx q1[4], q0[0];
rz(2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(-pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-2.4733252) q0[0];
rz(2.4710034) q1[4];
ry(pi/2) q1[4];
cx q1[3], q1[4];
rz(pi/4) q1[4];
cx q1[3], q1[4];
h q1[3];
rz(-pi/4) q1[4];
cx q1[2], q1[4];
rz(pi/8) q1[4];
cx q1[2], q1[4];
rz(-pi/4) q1[2];
cx q1[2], q1[3];
rz(pi/4) q1[3];
cx q1[2], q1[3];
h q1[2];
rz(-pi/4) q1[3];
rz(-pi/8) q1[4];
cx q1[1], q1[4];
rz(pi/16) q1[4];
cx q1[1], q1[4];
rz(-pi/8) q1[1];
cx q1[1], q1[3];
rz(pi/8) q1[3];
cx q1[1], q1[3];
rz(-pi/4) q1[1];
cx q1[1], q1[2];
rz(pi/4) q1[2];
cx q1[1], q1[2];
h q1[1];
rz(-pi/4) q1[2];
rz(-pi/8) q1[3];
rz(-pi/16) q1[4];
cx q1[0], q1[4];
rz(pi/32) q1[4];
cx q1[0], q1[4];
rz(-pi/16) q1[0];
cx q1[0], q1[3];
rz(pi/16) q1[3];
cx q1[0], q1[3];
rz(-pi/8) q1[0];
cx q1[0], q1[2];
rz(pi/8) q1[2];
cx q1[0], q1[2];
rz(-pi/4) q1[0];
cx q1[0], q1[1];
rz(pi/4) q1[1];
cx q1[0], q1[1];
h q1[0];
rz(-pi/4) q1[1];
rz(-pi/8) q1[2];
rz(-pi/16) q1[3];
rz(-pi/32) q1[4];
ry(0.28967817) q2[0];
cx q1[4], q2[0];
ry(-0.07880704) q2[0];
cx q1[3], q2[0];
ry(-0.10745406) q2[0];
cx q1[4], q2[0];
ry(0.059433034) q2[0];
cx q1[2], q2[0];
ry(0.037086759) q2[0];
cx q1[4], q2[0];
ry(-0.11113425) q2[0];
cx q1[3], q2[0];
ry(-0.090469198) q2[0];
cx q1[4], q2[0];
ry(0.11644025) q2[0];
cx q1[1], q2[0];
ry(0.097611808) q2[0];
cx q1[4], q2[0];
ry(-0.09205678) q2[0];
cx q1[3], q2[0];
ry(-0.11154458) q2[0];
cx q1[4], q2[0];
ry(0.033985812) q2[0];
cx q1[2], q2[0];
ry(0.049624102) q2[0];
cx q1[4], q2[0];
ry(-0.10831791) q2[0];
cx q1[3], q2[0];
ry(-0.083772717) q2[0];
cx q1[4], q2[0];
ry(0.16223736) q2[0];
cx q1[0], q2[0];
ry(0.14683263) q2[0];
cx q1[4], q2[0];
ry(-0.084469198) q2[0];
cx q1[3], q2[0];
ry(-0.10841311) q2[0];
cx q1[4], q2[0];
ry(0.048240009) q2[0];
cx q1[2], q2[0];
ry(0.033623576) q2[0];
cx q1[4], q2[0];
ry(-0.11157749) q2[0];
cx q1[3], q2[0];
ry(-0.092239785) q2[0];
cx q1[4], q2[0];
ry(0.094908439) q2[0];
cx q1[1], q2[0];
ry(0.10838905) q2[0];
cx q1[4], q2[0];
ry(-0.090848564) q2[0];
cx q1[3], q2[0];
ry(-0.11118869) q2[0];
cx q1[4], q2[0];
ry(0.036333382) q2[0];
cx q1[2], q2[0];
ry(0.055353647) q2[0];
cx q1[4], q2[0];
ry(-0.107649) q2[0];
cx q1[3], q2[0];
ry(-0.080853948) q2[0];
cx q1[4], q2[0];
ry(0.20101829) q2[0];
cx q1[0], q2[0];
rx(-3*pi/4) q1[0];
ry(-pi/2) q1[0];
cx q1[0], q1[1];
rz(-pi/4) q1[1];
cx q1[0], q1[1];
rz(pi/8) q1[0];
cx q1[0], q1[2];
rz(5*pi/4) q1[1];
ry(pi/2) q1[1];
rz(pi/4) q1[1];
rz(-pi/8) q1[2];
cx q1[0], q1[2];
rz(pi/16) q1[0];
cx q1[0], q1[3];
rz(pi/8) q1[2];
cx q1[1], q1[2];
rz(-pi/4) q1[2];
cx q1[1], q1[2];
rz(pi/8) q1[1];
rz(5*pi/4) q1[2];
ry(pi/2) q1[2];
rz(pi/4) q1[2];
rz(-pi/16) q1[3];
cx q1[0], q1[3];
rz(pi/32) q1[0];
cx q1[0], q1[4];
rz(pi/16) q1[3];
cx q1[1], q1[3];
rz(-pi/8) q1[3];
cx q1[1], q1[3];
rz(pi/16) q1[1];
rz(pi/8) q1[3];
cx q1[2], q1[3];
rz(-pi/4) q1[3];
cx q1[2], q1[3];
rz(pi/8) q1[2];
rz(5*pi/4) q1[3];
ry(pi/2) q1[3];
rz(pi/4) q1[3];
rz(-pi/32) q1[4];
cx q1[0], q1[4];
rz(-0.6682675) q1[0];
rz(pi/32) q1[4];
cx q1[1], q1[4];
rz(-pi/16) q1[4];
cx q1[1], q1[4];
rz(-0.6682675) q1[1];
rz(pi/16) q1[4];
cx q1[2], q1[4];
rz(-pi/8) q1[4];
cx q1[2], q1[4];
rz(-0.6682675) q1[2];
rz(pi/8) q1[4];
cx q1[3], q1[4];
rz(-pi/4) q1[4];
cx q1[3], q1[4];
rz(-0.6682675) q1[3];
rz(-3*pi/4) q1[4];
ry(pi/2) q1[4];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[4];
cx q1[4], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[4], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[3];
cx q1[3], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[3], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
cx q1[2], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[2];
cx q1[2], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[2];
cx q1[2], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[2];
cx q1[2], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[2], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
cx q1[1], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[1], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
rz(-0.6682675) q1[1];
cx q1[1], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[1], q0[0];
ry(1.0108711) q0[0];
rz(-0.6682675) q0[0];
cx q1[0], q0[0];
rz(-2.4733252) q0[0];
ry(1.0108711) q0[0];
rz(pi) q0[0];
cx q1[0], q0[0];
ry(1.0108711) q0[0];
rz(0.90252883) q0[0];
rz(-1.5288845) q1[0];
ry(pi/2) q1[0];
rz(-6.1993617) q1[1];
ry(pi/2) q1[1];
rz(3.30924) q1[2];
ry(pi/2) q1[2];
rz(3.4768873) q1[3];
ry(pi/2) q1[3];
rz(3.8121819) q1[4];
ry(pi/2) q1[4];
barrier q0[0], q1[0], q1[1], q1[2], q1[3], q1[4], q2[0];
meas[0] = measure q0[0];
meas[1] = measure q1[0];
meas[2] = measure q1[1];
meas[3] = measure q1[2];
meas[4] = measure q1[3];
meas[5] = measure q1[4];
meas[6] = measure q2[0];
