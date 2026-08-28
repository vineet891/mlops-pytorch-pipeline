# Assignment 3 write-up

**Roll number:** DA25G524
**Repository:** https://github.com/vineet891/mlops-pytorch-pipeline

## What was the most challenging part?

The hardest part was not the PyTorch model. It was getting a
production-style path to run on a company laptop whose network and
Kubernetes node did not match the assumptions in the assignment PDF.

Training and serving themselves were straightforward. A ResNet-18 stem
was changed from a 7x7 stride-2 convolution to a 3x3 stride-1
convolution so that 32x32 CIFAR-10 images are not destroyed before the
residual blocks. Metrics were logged as JSON lines. The serving image
runs as uid 10001, exposes port 8080, and uses GET /health for Docker
HEALTHCHECK and for Kubernetes probes. Those pieces behaved as designed
once they had data, a checkpoint, and enough CPU.

The first real friction was TLS. pip could install packages from PyPI
with a trusted-host workaround, but download.pytorch.org and
cs.toronto.edu both failed certificate verification inside Linux
containers. That is a TLS-inspecting proxy, not a bug in torchvision.
CIFAR-10 was therefore downloaded on the host with curl (which uses the
macOS Keychain), copied into a PersistentVolumeClaim with kubectl cp,
and the ConfigMap set download to false. The training Job then read
data that was already on the volume.

The second friction was scheduling. local-path uses
WaitForFirstConsumer, so PVCs stay Pending until a pod uses them. That
looked like a storage failure until a consumer pod was applied. The
Job then stayed Pending because the Rancher Desktop VM was too small
for the required 2 CPU / 4Gi requests. Raising the VM to 4 vCPUs and
8 GB let the same manifest schedule without weakening the resource
fields the PDF asks for. kubectl describe shows those limits, the
ConfigMap mount at /app/configs, and both PVC mounts.

A smaller but real issue was NumPy 2.x with torch 2.2.2
(_ARRAY_API not found). Pinning numpy==1.26.4 fixed ToTensor in tests
and in the serving container.

Once those environment problems were handled, the pipeline was
ordinary: a Kubernetes Job wrote classifier_v1.pt onto a PVC, a
two-replica Deployment mounted that claim read-only, and a ClusterIP
service on port 80 forwarded to container 8080. A short train (two
epochs, 5% of CIFAR-10) is enough to prove the path; it is not enough
for a strong accuracy number, and a dummy PNG is classified with
inflated confidence. The deliverable is the lifecycle, not the score.

Course staff confirmed that four pull requests in a short window are
acceptable, that pvc.yaml may be a separate file, that the GPU bonus
belongs in its own manifest, and that HPA and CI were optional. HPA
was skipped. CI and tests were still added because they catch config
and model-shape mistakes before a cluster apply.
