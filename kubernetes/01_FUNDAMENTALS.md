# Kubernetes Fundamentals

Container orchestration explained without the enterprise jargon.

---

## Table of Contents

1. [What Problem Does K8s Solve?](#what-problem-does-k8s-solve)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [Workload Resources](#workload-resources)
5. [Networking](#networking)
6. [Storage](#storage)
7. [Configuration](#configuration)
8. [How It All Fits Together](#how-it-all-fits-together)

---

## What Problem Does K8s Solve?

**Without Kubernetes:**
```
"Deploy my app on 10 servers"

Manual tasks:
- SSH into each server
- Pull new container image
- Stop old container, start new one
- Hope nothing breaks
- If server dies, manually move app
- Scale up? Manually add servers
- Load balancing? Configure separately
```

**With Kubernetes:**
```
"Deploy my app with 10 replicas"

kubectl apply -f deployment.yaml

K8s handles:
✓ Scheduling containers across nodes
✓ Restarting failed containers
✓ Rolling updates with zero downtime
✓ Auto-scaling based on load
✓ Service discovery and load balancing
✓ Storage management
✓ Secret management
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      CONTROL PLANE                              │ │
│  │                                                                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │ │
│  │  │ API Server   │  │   etcd       │  │ Controller Manager    │ │ │
│  │  │              │  │ (key-value   │  │ (runs control loops)  │ │ │
│  │  │ (REST API,   │  │  store for   │  │                       │ │ │
│  │  │  all comms   │  │  all cluster │  │ • Node controller     │ │ │
│  │  │  go through  │  │  state)      │  │ • Replication ctrl    │ │ │
│  │  │  here)       │  │              │  │ • Endpoint controller │ │ │
│  │  └──────────────┘  └──────────────┘  └───────────────────────┘ │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ Scheduler                                                 │  │ │
│  │  │ (decides which node runs which pod)                       │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                         WORKER NODES                            │ │
│  │                                                                 │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐            │ │
│  │  │      Node 1         │    │      Node 2         │            │ │
│  │  │  ┌───────────────┐  │    │  ┌───────────────┐  │            │ │
│  │  │  │    kubelet    │  │    │  │    kubelet    │  │            │ │
│  │  │  │ (node agent)  │  │    │  │ (node agent)  │  │            │ │
│  │  │  └───────────────┘  │    │  └───────────────┘  │            │ │
│  │  │  ┌───────────────┐  │    │  ┌───────────────┐  │            │ │
│  │  │  │  kube-proxy   │  │    │  │  kube-proxy   │  │            │ │
│  │  │  │ (networking)  │  │    │  │ (networking)  │  │            │ │
│  │  │  └───────────────┘  │    │  └───────────────┘  │            │ │
│  │  │  ┌───────────────┐  │    │  ┌───────────────┐  │            │ │
│  │  │  │   Container   │  │    │  │   Container   │  │            │ │
│  │  │  │   Runtime     │  │    │  │   Runtime     │  │            │ │
│  │  │  └───────────────┘  │    │  └───────────────┘  │            │ │
│  │  │                     │    │                     │            │ │
│  │  │  [Pod] [Pod] [Pod]  │    │  [Pod] [Pod]        │            │ │
│  │  └─────────────────────┘    └─────────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| **API Server** | Front door to the cluster. All kubectl commands go here. |
| **etcd** | Stores all cluster state. The "database" of K8s. |
| **Scheduler** | Watches for new pods, assigns them to nodes. |
| **Controller Manager** | Runs loops that maintain desired state. |
| **kubelet** | Agent on each node. Runs and monitors pods. |
| **kube-proxy** | Manages network rules for pod communication. |

---

## Core Concepts

### The Declarative Model

You describe **what you want**, not how to get there:

```yaml
# "I want 3 copies of my app running"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3           # Desired state
  selector:
    matchLabels:
      app: my-app
  template:
    spec:
      containers:
      - name: my-app
        image: my-app:v1
```

K8s continuously works to make **actual state = desired state**.

### Labels and Selectors

Labels are key-value pairs for organizing resources:

```yaml
metadata:
  labels:
    app: frontend
    environment: production
    team: checkout
```

Selectors find resources by labels:

```yaml
selector:
  matchLabels:
    app: frontend
    environment: production
```

**Use cases:**
- Services find their pods
- Deployments manage their pods
- Queries: `kubectl get pods -l app=frontend`

### Namespaces

Virtual clusters within a cluster:

```
┌─────────────────────────────────────────────────────┐
│ Cluster                                             │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ default     │  │ production  │  │ staging     │ │
│  │             │  │             │  │             │ │
│  │  [pods]     │  │  [pods]     │  │  [pods]     │ │
│  │  [services] │  │  [services] │  │  [services] │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐                  │
│  │ kube-system │  │ monitoring  │                  │
│  │ (K8s stuff) │  │             │                  │
│  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────┘
```

```bash
# Work in a namespace
kubectl get pods -n production
kubectl config set-context --current --namespace=production
```

---

## Workload Resources

### Pod

**The smallest deployable unit.** One or more containers that share:
- Network namespace (same IP, can use localhost)
- Storage volumes
- Lifecycle

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: nginx:1.21
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

**You rarely create pods directly.** Use Deployments instead.

### Deployment

Manages pods with:
- Desired replica count
- Rolling updates
- Rollback capability

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
```

**What happens:**
1. Deployment creates a ReplicaSet
2. ReplicaSet creates 3 Pods
3. If a pod dies, ReplicaSet creates a new one
4. Update image? New ReplicaSet, gradual rollover

```
Deployment
    │
    └──▶ ReplicaSet (v1) ──▶ Pod, Pod, Pod
    │
    └──▶ ReplicaSet (v2) ──▶ (created on update)
```

### StatefulSet

For stateful applications (databases, message queues):

```
Deployment Pods:          StatefulSet Pods:
┌───────────────────┐     ┌───────────────────┐
│ Random names:     │     │ Stable names:     │
│ nginx-7d8f9c-abc  │     │ mysql-0           │
│ nginx-7d8f9c-def  │     │ mysql-1           │
│ nginx-7d8f9c-ghi  │     │ mysql-2           │
│                   │     │                   │
│ Interchangeable   │     │ Ordered startup   │
│ No stable storage │     │ Stable storage    │
│ No stable network │     │ Stable DNS name   │
└───────────────────┘     └───────────────────┘
```

### DaemonSet

Run one pod per node (for node-level services):

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: log-collector
spec:
  selector:
    matchLabels:
      name: log-collector
  template:
    # ... pod template
```

**Use cases:** Log collectors, monitoring agents, network plugins.

### Job / CronJob

**Job:** Run to completion (batch processing)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: migration-tool:v1
      restartPolicy: Never
  backoffLimit: 4
```

**CronJob:** Scheduled jobs

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:v1
          restartPolicy: OnFailure
```

### Resource Comparison

| Resource | Use Case | Replicas | Pod Names |
|----------|----------|----------|-----------|
| **Deployment** | Stateless apps | N replicas, interchangeable | Random |
| **StatefulSet** | Databases, stateful apps | Ordered, stable identity | Indexed (app-0, app-1) |
| **DaemonSet** | Per-node agents | One per node | Per-node |
| **Job** | Batch processing | Run to completion | Random |
| **CronJob** | Scheduled tasks | Periodic jobs | Random |

---

## Networking

### Service

Stable network endpoint for a set of pods:

```
Without Service:                With Service:
┌─────────────────────┐        ┌─────────────────────┐
│ Pods come and go    │        │  Service            │
│                     │        │  ┌───────────────┐  │
│ Pod: 10.0.1.5  ✗    │        │  │ my-service    │  │
│ Pod: 10.0.1.8  ✓    │        │  │ 10.96.0.1     │  │
│ Pod: 10.0.1.12 ✓    │        │  │               │  │
│                     │        │  │    ┌─────┐    │  │
│ How do clients      │        │  │    │ LB  │    │  │
│ know which IP?      │        │  │    └──┬──┘    │  │
└─────────────────────┘        │  └──────┼────────┘  │
                               │         │           │
                               │    ┌────┴────┐      │
                               │    ▼    ▼    ▼      │
                               │  [Pod] [Pod] [Pod]  │
                               └─────────────────────┘
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app       # Find pods with this label
  ports:
  - port: 80          # Service port
    targetPort: 8080  # Pod port
  type: ClusterIP     # Internal only (default)
```

### Service Types

| Type | Accessibility | Use Case |
|------|---------------|----------|
| **ClusterIP** | Internal only | Service-to-service |
| **NodePort** | External via node IP:port | Dev/testing |
| **LoadBalancer** | External via cloud LB | Production |
| **ExternalName** | DNS alias | External services |

```yaml
# LoadBalancer example (creates cloud LB)
apiVersion: v1
kind: Service
metadata:
  name: my-public-service
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
```

### Ingress

Layer 7 (HTTP) routing:

```
                    ┌─────────────────────────────────────┐
Internet ──────────▶│           Ingress Controller        │
                    │                                     │
                    │  api.example.com ──▶ api-service    │
                    │  www.example.com ──▶ web-service    │
                    │  example.com/api ──▶ api-service    │
                    └─────────────────────────────────────┘
```

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
  - host: www.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

### DNS in Kubernetes

Every service gets a DNS name:

```
<service-name>.<namespace>.svc.cluster.local

Examples:
  my-service.default.svc.cluster.local
  database.production.svc.cluster.local
  
Shorthand (same namespace):
  my-service
  
Shorthand (different namespace):
  database.production
```

---

## Storage

### Volumes

Attach storage to pods:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    emptyDir: {}     # Temporary, deleted with pod
```

### Volume Types

| Type | Persistence | Use Case |
|------|-------------|----------|
| **emptyDir** | Pod lifetime | Temp files, cache |
| **hostPath** | Node lifetime | Node-level data |
| **configMap** | Config changes | Config files |
| **secret** | Config changes | Sensitive config |
| **persistentVolumeClaim** | Persistent | Databases, uploads |

### PersistentVolume (PV) & PersistentVolumeClaim (PVC)

Decouple storage provisioning from usage:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Admin creates PV              Dev creates PVC             │
│   (or dynamic provisioning)     (requests storage)          │
│                                                             │
│   ┌─────────────────┐          ┌─────────────────┐         │
│   │ PersistentVolume│◀─ binds ─│ PersistentVolume│         │
│   │                 │          │ Claim            │         │
│   │ 100Gi SSD       │          │ "I need 50Gi"   │         │
│   │ AWS EBS         │          │                 │         │
│   └─────────────────┘          └────────┬────────┘         │
│                                         │                   │
│                                         ▼                   │
│                                    ┌─────────┐              │
│                                    │   Pod   │              │
│                                    │ (mounts │              │
│                                    │  PVC)   │              │
│                                    └─────────┘              │
└─────────────────────────────────────────────────────────────┘
```

```yaml
# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Use default storage class
---
# Pod using the PVC
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: my-storage
```

### Access Modes

| Mode | Abbreviation | Description |
|------|--------------|-------------|
| ReadWriteOnce | RWO | Single node read-write |
| ReadOnlyMany | ROX | Multiple nodes read-only |
| ReadWriteMany | RWX | Multiple nodes read-write |

---

## Configuration

### ConfigMap

Non-sensitive configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "postgres.default.svc"
  LOG_LEVEL: "info"
  config.json: |
    {
      "feature_flags": {
        "new_ui": true
      }
    }
```

**Use in pod:**

```yaml
# As environment variables
env:
- name: DATABASE_HOST
  valueFrom:
    configMapKeyRef:
      name: app-config
      key: DATABASE_HOST

# As mounted file
volumeMounts:
- name: config
  mountPath: /etc/config
volumes:
- name: config
  configMap:
    name: app-config
```

### Secret

Sensitive data (base64 encoded, not encrypted by default!):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  username: YWRtaW4=      # base64 of "admin"
  password: cGFzc3dvcmQ=  # base64 of "password"
```

```bash
# Create from literals
kubectl create secret generic db-creds \
  --from-literal=username=admin \
  --from-literal=password=secret

# Create from file
kubectl create secret generic tls-cert \
  --from-file=cert.pem \
  --from-file=key.pem
```

**Use in pod:**

```yaml
env:
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: db-credentials
      key: password
```

---

## How It All Fits Together

### Example: Complete Web Application

```yaml
# 1. ConfigMap for app settings
apiVersion: v1
kind: ConfigMap
metadata:
  name: webapp-config
data:
  API_URL: "http://api-service:8080"
---
# 2. Secret for sensitive data
apiVersion: v1
kind: Secret
metadata:
  name: webapp-secrets
type: Opaque
stringData:
  API_KEY: "super-secret-key"
---
# 3. Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: webapp
        image: myapp:v1
        ports:
        - containerPort: 3000
        env:
        - name: API_URL
          valueFrom:
            configMapKeyRef:
              name: webapp-config
              key: API_URL
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: webapp-secrets
              key: API_KEY
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 3
---
# 4. Service (internal)
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
spec:
  selector:
    app: webapp
  ports:
  - port: 80
    targetPort: 3000
---
# 5. Ingress (external)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: tls-secret
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: webapp-service
            port:
              number: 80
```

### Request Flow

```
1. User → myapp.example.com
       ↓
2. DNS → Ingress Controller IP
       ↓
3. Ingress Controller → matches host rule
       ↓
4. Routes to webapp-service:80
       ↓
5. Service → load balances to one of 3 pods
       ↓
6. Pod container:3000 handles request
```

---

## Quick Reference

### Resource Hierarchy

```
Cluster
 └── Namespace
      ├── Deployment / StatefulSet / DaemonSet
      │    └── ReplicaSet (managed by Deployment)
      │         └── Pod
      │              └── Container
      ├── Service
      ├── Ingress
      ├── ConfigMap
      ├── Secret
      └── PersistentVolumeClaim
```

### Common kubectl Commands

```bash
# Basics
kubectl get pods                    # List pods
kubectl get all                     # List common resources
kubectl describe pod <name>         # Detailed info
kubectl logs <pod>                  # View logs
kubectl exec -it <pod> -- /bin/sh  # Shell into pod

# Apply/Delete
kubectl apply -f manifest.yaml      # Create/update
kubectl delete -f manifest.yaml     # Delete

# Debugging
kubectl get events                  # Cluster events
kubectl top pods                    # Resource usage
kubectl port-forward <pod> 8080:80  # Local access
```

---

Next: [Practical Kubernetes](./02_PRACTICAL.md) - Commands, debugging, and real-world patterns
