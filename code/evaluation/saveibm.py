from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="eYYOxvJ4KF8Pxmp-B_HscxAqZfnYgkmga8caz7QyJj0n",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/e74637527db84204a3dfe17cb9966ebf:a65a8942-414b-4907-a5e9-9d68bd51e634::",
)

print("Saved IBM Runtime account")
