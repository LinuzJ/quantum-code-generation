OPENQASM 3.0;
include "stdgates.inc";
bit[1] c0;
qubit[41] q0;
ry(1.704442104973648) q0[1];
ry(1.0958779629585713) q0[2];
ry(1.230977370949048) q0[3];
ry(0.48760212662589236) q0[4];
ry(0.6150967398502879) q0[5];
ry(1.8362803692692948) q0[6];
ry(0.17035702146686962) q0[7];
ry(0.0760181161010892) q0[8];
ry(0.9048445541342406) q0[9];
ry(0.35629343147681364) q0[10];
ry(1.199395253530262) q0[11];
ry(0.2540130613766632) q0[12];
ry(0.4796584847365919) q0[13];
ry(0.12412549084739496) q0[14];
ry(0.8762899323374341) q0[15];
ry(2.076321021715302) q0[16];
ry(0.459013454479424) q0[17];
ry(0.6772913953916866) q0[18];
ry(2.50357175527731) q0[19];
ry(0.9517094639363597) q0[20];
ry(0.8549329989283637) q0[21];
ry(2.5377786539658453) q0[22];
ry(1.7272597199680724) q0[23];
ry(0.6329632976374331) q0[24];
ry(3.1344638057987977) q0[25];
ry(3.0412791505597867) q0[26];
ry(2.5111584596518126) q0[27];
ry(0.8477076457109352) q0[28];
ry(0.7177241698504794) q0[29];
ry(1.2665404563869145) q0[30];
ry(1.3097989599488284) q0[31];
ry(2.1652614066027747) q0[32];
ry(0.07153878230580002) q0[33];
ry(2.725902017084623) q0[34];
ry(2.8993704288649926) q0[35];
ry(2.543292869372673) q0[36];
ry(1.0366165973566377) q0[37];
ry(1.67501464903548) q0[38];
ry(0.8493287435327369) q0[39];
ry(1.4843570631924323) q0[40];
h q0[0];
cswap q0[0], q0[1], q0[21];
cswap q0[0], q0[2], q0[22];
cswap q0[0], q0[3], q0[23];
cswap q0[0], q0[4], q0[24];
cswap q0[0], q0[5], q0[25];
cswap q0[0], q0[6], q0[26];
cswap q0[0], q0[7], q0[27];
cswap q0[0], q0[8], q0[28];
cswap q0[0], q0[9], q0[29];
cswap q0[0], q0[10], q0[30];
cswap q0[0], q0[11], q0[31];
cswap q0[0], q0[12], q0[32];
cswap q0[0], q0[13], q0[33];
cswap q0[0], q0[14], q0[34];
cswap q0[0], q0[15], q0[35];
cswap q0[0], q0[16], q0[36];
cswap q0[0], q0[17], q0[37];
cswap q0[0], q0[18], q0[38];
cswap q0[0], q0[19], q0[39];
cswap q0[0], q0[20], q0[40];
h q0[0];
c0[0] = measure q0[0];
