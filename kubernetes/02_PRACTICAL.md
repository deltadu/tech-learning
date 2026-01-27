# Practical Kubernetes

Essential commands, debugging techniques, and real-world patterns.

---

## Table of Contents

1. [Essential kubectl Commands](#essential-kubectl-commands)
2. [Debugging Pods](#debugging-pods)
3. [Resource Management](#resource-management)
4. [Deployment Strategies](#deployment-strategies)
5. [Health Checks](#health-checks)
6. [Scaling](#scaling)
7. [Networking Debugging](#networking-debugging)
8. [Security Basics](#security-basics)
9. [Helm Basics](#helm-basics)
10. [Common Patterns](#common-patterns)
11. [Gotchas & Tips](#gotchas--tips)

---

## Essential kubectl Commands

### Getting Information

```bash
# Cluster info
kubectl cluster-info
kubectl get nodes
kubectl top nodes                    # Node resource usage

# List resources
kubectl get pods                     # Current namespace
kubectl get pods -A                  # All namespaces
kubectl get pods -o wide             # More columns (node, IP)
kubectl get pods -w                  # Watch for changes

# Multiple resource types
kubectl get all                      # Common resources
kubectl get pods,services,deployments

# Filter by label
kubectl get pods -l app=nginx
kubectl get pods -l 'app in (nginx,apache)'

# Output formats
kubectl get pods -o yaml             # Full YAML
kubectl get pods -o json             # Full JSON
kubectl get pods -o name             # Just names
kubectl get pods -o custom-columns=NAME:.metadata.name,STATUS:.status.phase
```

### Describing Resources

```bash
# Detailed info (events, conditions, etc.)
kubectl describe pod <pod-name>
kubectl describe deployment <name>
kubectl describe node <node-name>

# Events (great for debugging)
kubectl get events --sort-by='.lastTimestamp'
kubectl get events --field-selector type=Warning
```

### Creating and Modifying

```bash
# Apply manifest
kubectl apply -f manifest.yaml
kubectl apply -f ./manifests/        # Entire directory
kubectl apply -f https://url/manifest.yaml

# Create imperatively (for quick tests)
kubectl run nginx --image=nginx
kubectl create deployment nginx --image=nginx
kubectl expose deployment nginx --port=80 --type=LoadBalancer

# Edit running resource
kubectl edit deployment nginx
kubectl set image deployment/nginx nginx=nginx:1.22

# Delete
kubectl delete pod <pod-name>
kubectl delete -f manifest.yaml
kubectl delete pods -l app=test      # By label
kubectl delete pods --all            # All pods in namespace (!)
```

### Interacting with Pods

```bash
# Logs
kubectl logs <pod-name>
kubectl logs <pod-name> -c <container>  # Specific container
kubectl logs <pod-name> --previous      # Previous instance (after crash)
kubectl logs -f <pod-name>              # Follow/tail
kubectl logs -l app=nginx               # By label

# Execute commands
kubectl exec <pod-name> -- ls /
kubectl exec -it <pod-name> -- /bin/sh  # Interactive shell
kubectl exec -it <pod-name> -c <container> -- /bin/sh

# Copy files
kubectl cp <pod-name>:/path/file ./local-file
kubectl cp ./local-file <pod-name>:/path/file

# Port forward
kubectl port-forward pod/<pod-name> 8080:80
kubectl port-forward svc/<service-name> 8080:80
kubectl port-forward deployment/<name> 8080:80
```

### Context and Config

```bash
# View config
kubectl config view
kubectl config current-context

# Switch context (cluster/user)
kubectl config use-context my-cluster

# Switch namespace
kubectl config set-context --current --namespace=production

# Create alias for namespace
alias kprod='kubectl -n production'
```

---

## Debugging Pods

### Pod Not Starting?

**Check status:**
```bash
kubectl get pods
kubectl describe pod <pod-name>
```

**Common states:**

| Status | Meaning | Check |
|--------|---------|-------|
| `Pending` | Can't be scheduled | Resources? Node selector? |
| `ContainerCreating` | Image pulling or volume mounting | Image exists? Pull secret? |
| `ImagePullBackOff` | Can't pull image | Image name? Registry auth? |
| `CrashLoopBackOff` | Container keeps crashing | Logs! App error? |
| `Error` | Container exited with error | Logs! |
| `OOMKilled` | Out of memory | Increase memory limit |

### Debug Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Pod Not Running - Debug Flow                                 │
│                                                              │
│ 1. kubectl get pods                                          │
│    └─▶ Check STATUS column                                   │
│                                                              │
│ 2. kubectl describe pod <name>                               │
│    └─▶ Look at:                                              │
│        • Events (bottom of output)                           │
│        • Conditions                                          │
│        • State/Last State                                    │
│                                                              │
│ 3. kubectl logs <pod> [--previous]                           │
│    └─▶ Application errors?                                   │
│                                                              │
│ 4. kubectl get events --sort-by='.lastTimestamp'             │
│    └─▶ Cluster-level issues?                                 │
│                                                              │
│ 5. kubectl exec -it <pod> -- /bin/sh                         │
│    └─▶ Check filesystem, env vars, connectivity              │
└──────────────────────────────────────────────────────────────┘
```

### Common Fixes

**ImagePullBackOff:**
```bash
# Check image name
kubectl describe pod <name> | grep Image

# Check pull secret
kubectl get secrets
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<pass>
```

**CrashLoopBackOff:**
```bash
# Check logs from crashed container
kubectl logs <pod> --previous

# Common causes:
# - App crashes immediately (check CMD/ENTRYPOINT)
# - Missing config/env vars
# - Can't connect to dependency
# - OOM (check memory limits)
```

**Pending (Insufficient resources):**
```bash
kubectl describe pod <name> | grep -A5 Events
# Look for "Insufficient cpu" or "Insufficient memory"

# Check node capacity
kubectl describe nodes | grep -A5 "Allocated resources"
```

### Debug Container

Run a debug container in the same namespace:

```bash
# Run a debug pod
kubectl run debug --rm -it --image=busybox -- /bin/sh

# Or with more tools
kubectl run debug --rm -it --image=nicolaka/netshoot -- /bin/bash

# Debug inside (curl, nslookup, ping, etc.)
nslookup my-service
curl http://my-service:80
```

---

## Resource Management

### Resource Requests and Limits

```yaml
resources:
  requests:           # Guaranteed minimum
    cpu: "100m"       # 0.1 CPU cores
    memory: "128Mi"   # 128 megabytes
  limits:             # Maximum allowed
    cpu: "500m"
    memory: "256Mi"
```

**CPU units:**
- `1` = 1 vCPU/core
- `100m` = 0.1 CPU (100 millicores)
- `500m` = 0.5 CPU

**Memory units:**
- `128Mi` = 128 mebibytes (134MB)
- `1Gi` = 1 gibibyte

**What happens when limits are hit:**

| Resource | When exceeded |
|----------|---------------|
| CPU | Throttled (slowed down) |
| Memory | Container killed (OOMKilled) |

### LimitRange

Default limits for a namespace:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: production
spec:
  limits:
  - default:
      cpu: "500m"
      memory: "256Mi"
    defaultRequest:
      cpu: "100m"
      memory: "128Mi"
    type: Container
```

### ResourceQuota

Limit total resources in namespace:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: namespace-quota
  namespace: development
spec:
  hard:
    requests.cpu: "10"
    requests.memory: "20Gi"
    limits.cpu: "20"
    limits.memory: "40Gi"
    pods: "50"
```

---

## Deployment Strategies

### Rolling Update (Default)

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%        # Extra pods during update
      maxUnavailable: 25%  # Pods that can be down
```

```
Old pods:  [v1] [v1] [v1] [v1]
           ─────────────────────▶
Update:    [v1] [v1] [v1] [v2]   Start v2
           [v1] [v1] [v2] [v2]   More v2
           [v1] [v2] [v2] [v2]   Almost done
           [v2] [v2] [v2] [v2]   Complete
```

### Recreate

Kill all old pods, then create new:

```yaml
spec:
  strategy:
    type: Recreate
```

**Use when:** App can't run multiple versions simultaneously.

### Blue-Green

Two deployments, switch service:

```yaml
# Blue deployment (current)
metadata:
  name: app-blue
  labels:
    version: blue

# Green deployment (new)
metadata:
  name: app-green
  labels:
    version: green

# Service - switch selector to change traffic
spec:
  selector:
    app: myapp
    version: blue   # Change to 'green' to switch
```

### Canary

Gradually shift traffic:

```yaml
# Stable deployment (90%)
spec:
  replicas: 9
  template:
    metadata:
      labels:
        app: myapp
        version: stable

# Canary deployment (10%)
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: myapp
        version: canary

# Service selects both (by app label only)
spec:
  selector:
    app: myapp  # Matches both versions
```

### Rollback

```bash
# View history
kubectl rollout history deployment/myapp

# Rollback to previous
kubectl rollout undo deployment/myapp

# Rollback to specific revision
kubectl rollout undo deployment/myapp --to-revision=2

# Check status
kubectl rollout status deployment/myapp
```

---

## Health Checks

### Liveness Probe

"Is the container alive?" If fails, container is restarted.

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 30    # Wait before first check
  periodSeconds: 10          # Check every 10s
  timeoutSeconds: 5          # Timeout per check
  failureThreshold: 3        # Restart after 3 failures
```

### Readiness Probe

"Is the container ready for traffic?" If fails, removed from service endpoints.

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3
```

### Startup Probe

For slow-starting containers. Disables liveness/readiness until passes.

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  failureThreshold: 30       # 30 * 10s = 5 min to start
  periodSeconds: 10
```

### Probe Types

```yaml
# HTTP GET
httpGet:
  path: /health
  port: 8080

# TCP Socket (just checks port is open)
tcpSocket:
  port: 3306

# Exec command (success = exit code 0)
exec:
  command:
  - cat
  - /tmp/healthy
```

---

## Scaling

### Manual Scaling

```bash
kubectl scale deployment myapp --replicas=5
```

### Horizontal Pod Autoscaler (HPA)

Scale based on metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70   # Scale when CPU > 70%
```

```bash
# Create HPA quickly
kubectl autoscale deployment myapp --min=2 --max=10 --cpu-percent=70

# Check HPA status
kubectl get hpa
```

### Vertical Pod Autoscaler (VPA)

Adjust resource requests/limits (requires VPA installed):

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: myapp-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  updatePolicy:
    updateMode: "Auto"  # or "Off" for recommendations only
```

---

## Networking Debugging

### DNS Issues

```bash
# Run debug pod
kubectl run debug --rm -it --image=busybox -- /bin/sh

# Test DNS
nslookup kubernetes.default
nslookup my-service.my-namespace.svc.cluster.local

# Check DNS pods
kubectl get pods -n kube-system -l k8s-app=kube-dns
```

### Service Not Reachable

```bash
# Check service exists and has endpoints
kubectl get svc my-service
kubectl get endpoints my-service

# No endpoints? Check selector matches pod labels
kubectl get pods --show-labels
kubectl describe svc my-service | grep Selector

# Test from within cluster
kubectl run curl --rm -it --image=curlimages/curl -- \
  curl http://my-service:80
```

### Network Policies

Check if network policies are blocking traffic:

```bash
# List network policies
kubectl get networkpolicies -A

# Describe policy
kubectl describe networkpolicy <name>
```

---

## Security Basics

### RBAC (Role-Based Access Control)

**Role** (namespaced) or **ClusterRole** (cluster-wide):

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
```

**RoleBinding** or **ClusterRoleBinding**:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### Security Context

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: app
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
```

### Pod Security Standards

Labels on namespace to enforce security:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

---

## Helm Basics

### What is Helm?

Package manager for Kubernetes. Charts = packages of K8s manifests.

```bash
# Install Helm
brew install helm

# Add repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Search charts
helm search repo nginx
helm search hub nginx  # Search Artifact Hub

# Install chart
helm install my-release bitnami/nginx

# Install with values
helm install my-release bitnami/nginx \
  --set replicaCount=3 \
  --set service.type=LoadBalancer

# Or with values file
helm install my-release bitnami/nginx -f values.yaml
```

### Managing Releases

```bash
# List releases
helm list
helm list -A               # All namespaces

# Upgrade
helm upgrade my-release bitnami/nginx --set replicaCount=5

# Rollback
helm rollback my-release 1

# Uninstall
helm uninstall my-release

# See what would be installed (dry run)
helm install my-release bitnami/nginx --dry-run

# Get values
helm get values my-release
helm show values bitnami/nginx   # Default values
```

---

## Common Patterns

### Sidecar Container

Multiple containers in one pod:

```yaml
spec:
  containers:
  - name: app
    image: myapp:v1
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
  - name: log-shipper           # Sidecar
    image: fluentd
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
  volumes:
  - name: logs
    emptyDir: {}
```

### Init Container

Run before main containers:

```yaml
spec:
  initContainers:
  - name: wait-for-db
    image: busybox
    command: ['sh', '-c', 
      'until nc -z database 5432; do sleep 2; done']
  containers:
  - name: app
    image: myapp:v1
```

### Jobs for Database Migrations

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migrate
  annotations:
    "helm.sh/hook": pre-upgrade
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: myapp:v1
        command: ["./migrate.sh"]
      restartPolicy: Never
  backoffLimit: 3
```

### External Secrets

Don't store secrets in Git. Use external secrets operators:

```yaml
# With External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: db-credentials
  data:
  - secretKey: password
    remoteRef:
      key: production/database
      property: password
```

---

## Gotchas & Tips

### 1. Resources Are Required in Production

```yaml
# Always set requests and limits
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "500m"
    memory: "256Mi"
```

Without them:
- Pods can be evicted unpredictably
- One pod can starve others
- Autoscaling won't work

### 2. Use Specific Image Tags

```yaml
# ❌ Bad
image: myapp:latest

# ✓ Good
image: myapp:v1.2.3
image: myapp@sha256:abc123...  # Even better
```

`latest` is unpredictable and breaks rollbacks.

### 3. Configure Probes Properly

```yaml
# ❌ No probes = K8s can't know if app is healthy

# ❌ Too aggressive
livenessProbe:
  initialDelaySeconds: 0    # App hasn't started!
  periodSeconds: 1          # Too frequent

# ✓ Good
livenessProbe:
  initialDelaySeconds: 30   # Give app time to start
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3       # Don't kill on first failure
```

### 4. Graceful Shutdown

```yaml
spec:
  terminationGracePeriodSeconds: 60  # Default is 30
  containers:
  - name: app
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 10"]
```

Handle SIGTERM in your app to finish current requests.

### 5. Don't Use Default Namespace

```bash
# Create namespace for your app
kubectl create namespace myapp

# Always specify namespace
kubectl apply -f manifest.yaml -n myapp
```

### 6. Pod Disruption Budget

Prevent too many pods going down during node maintenance:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp-pdb
spec:
  minAvailable: 2        # Or maxUnavailable: 1
  selector:
    matchLabels:
      app: myapp
```

### 7. Anti-Affinity for High Availability

Spread pods across nodes:

```yaml
spec:
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: myapp
          topologyKey: kubernetes.io/hostname
```

---

## Quick Reference

### Pod Status Meanings

| Status | Meaning |
|--------|---------|
| Pending | Waiting to be scheduled |
| ContainerCreating | Pulling image, setting up |
| Running | At least one container running |
| Succeeded | All containers completed successfully |
| Failed | All containers terminated, at least one failed |
| Unknown | Node communication lost |

### Useful Aliases

```bash
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get svc'
alias kgd='kubectl get deployments'
alias kga='kubectl get all'
alias kd='kubectl describe'
alias kl='kubectl logs'
alias ke='kubectl exec -it'
alias kaf='kubectl apply -f'
alias kdf='kubectl delete -f'
```

### Debug Commands Cheat Sheet

```bash
# Is it running?
kubectl get pods

# Why isn't it running?
kubectl describe pod <name>
kubectl get events --sort-by='.lastTimestamp'

# What's it outputting?
kubectl logs <pod> [-f] [--previous]

# What's inside?
kubectl exec -it <pod> -- /bin/sh

# Can I reach it?
kubectl port-forward pod/<pod> 8080:80
kubectl run curl --rm -it --image=curlimages/curl -- curl <url>

# What's the resource usage?
kubectl top pods
kubectl top nodes
```
