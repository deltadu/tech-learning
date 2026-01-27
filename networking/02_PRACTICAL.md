# Practical Networking

Network tricks and techniques useful for robotics and app development.

---

## Table of Contents

1. [Debugging Network Issues](#debugging-network-issues)
2. [Working with APIs](#working-with-apis)
3. [WebSockets & Real-Time Communication](#websockets--real-time-communication)
4. [Robotics Networking](#robotics-networking)
5. [Mobile App Networking](#mobile-app-networking)
6. [Network Security Basics](#network-security-basics)
7. [Performance Optimization](#performance-optimization)
8. [Common Gotchas](#common-gotchas)

---

## Debugging Network Issues

### The Debugging Hierarchy

When something doesn't work, check in this order:

```
1. Can you reach the internet at all?     → ping 8.8.8.8
2. Is DNS working?                        → ping google.com
3. Is the port open/listening?            → nc -zv host port
4. Is the service responding?             → curl http://host:port
5. Is it an application issue?            → check logs
```

### Essential Tools

#### curl - The Swiss Army Knife

```bash
# Basic GET
curl https://api.example.com/users

# POST with JSON
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John"}'

# With authentication
curl -H "Authorization: Bearer TOKEN" https://api.example.com/me

# See headers and timing
curl -v https://api.example.com         # verbose
curl -I https://api.example.com         # headers only
curl -w "@curl-format.txt" -o /dev/null -s https://example.com  # timing

# Follow redirects
curl -L https://short.url/abc

# Download file
curl -O https://example.com/file.zip

# Upload file
curl -F "file=@local.txt" https://api.example.com/upload
```

**Timing format file (curl-format.txt):**
```
    time_namelookup:  %{time_namelookup}s\n
       time_connect:  %{time_connect}s\n
    time_appconnect:  %{time_appconnect}s\n
   time_pretransfer:  %{time_pretransfer}s\n
      time_redirect:  %{time_redirect}s\n
 time_starttransfer:  %{time_starttransfer}s\n
                     ----------\n
         time_total:  %{time_total}s\n
```

#### netcat (nc) - TCP/UDP Swiss Army Knife

```bash
# Check if port is open
nc -zv google.com 443

# Listen on a port (simple server)
nc -l 8080

# Connect to a port (simple client)
nc localhost 8080

# Send UDP packet
echo "hello" | nc -u localhost 9999

# Port scanning
nc -zv host 20-100
```

#### tcpdump - Packet Capture

```bash
# Capture HTTP traffic
sudo tcpdump -i any port 80

# Capture with readable output
sudo tcpdump -i any -A port 80

# Capture to file for Wireshark
sudo tcpdump -i any -w capture.pcap

# Filter by host
sudo tcpdump host 192.168.1.100

# Filter by protocol
sudo tcpdump icmp
sudo tcpdump udp
```

#### ss / netstat - Connection Status

```bash
# See all listening ports
ss -tulpn                    # Linux
netstat -an | grep LISTEN    # macOS

# See established connections
ss -t state established

# See which process uses a port
lsof -i :8080
```

### Diagnosing Latency

```bash
# Check round-trip time
ping -c 10 google.com

# See network path
traceroute google.com        # ICMP
traceroute -T google.com     # TCP (better through firewalls)

# Continuous monitoring
mtr google.com               # combines ping + traceroute
```

---

## Working with APIs

### REST Best Practices

**Request design:**
```
GET    /users          → List users
GET    /users/123      → Get user 123
POST   /users          → Create user
PUT    /users/123      → Replace user 123
PATCH  /users/123      → Update user 123
DELETE /users/123      → Delete user 123

# Nested resources
GET    /users/123/posts     → User 123's posts
POST   /users/123/posts     → Create post for user 123
```

**Query parameters for filtering/pagination:**
```
GET /users?status=active&sort=created_at&order=desc
GET /users?page=2&limit=20
GET /users?fields=id,name,email    # sparse fieldsets
```

### Handling API Errors Gracefully

```python
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

def fetch_with_retry(url, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # raises HTTPError for 4xx/5xx
            return response.json()

        except Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # exponential backoff

        except HTTPError as e:
            if e.response.status_code == 429:  # rate limited
                retry_after = int(e.response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
            elif e.response.status_code >= 500:  # server error, retry
                time.sleep(2 ** attempt)
            else:  # client error, don't retry
                raise

        except RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

### Rate Limiting

Most APIs have rate limits. Handle them properly:

```python
import time
from functools import wraps

class RateLimiter:
    def __init__(self, calls_per_second):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

# Usage
limiter = RateLimiter(calls_per_second=10)

for item in items:
    limiter.wait()
    response = api.call(item)
```

### Connection Pooling

For high-throughput, reuse connections:

```python
import requests

# Session reuses TCP connections
session = requests.Session()

# Much faster than creating new connection each time
for url in urls:
    response = session.get(url)
```

---

## WebSockets & Real-Time Communication

### When to Use What

| Technology | Use Case | Latency | Complexity |
|------------|----------|---------|------------|
| HTTP Polling | Simple, low-frequency updates | High | Low |
| Long Polling | Moderate updates, fallback | Medium | Medium |
| Server-Sent Events (SSE) | One-way server→client | Low | Low |
| WebSocket | Bidirectional, real-time | Very Low | Medium |
| WebRTC | Peer-to-peer, video/audio | Lowest | High |

### WebSocket Basics

```
HTTP: Request → Response → Done
      Request → Response → Done
      (new connection each time)

WebSocket: Handshake (HTTP upgrade) → Persistent connection
           ←── messages ──→
           ←── messages ──→
           (same connection)
```

**Handshake:**
```
Client:
GET /chat HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==

Server:
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

### WebSocket Patterns

**Heartbeat/Ping-Pong:**
```python
# Keep connection alive through firewalls/proxies
async def heartbeat(ws, interval=30):
    while True:
        await asyncio.sleep(interval)
        await ws.ping()
```

**Reconnection with exponential backoff:**
```python
async def connect_with_retry(url):
    retry_delay = 1
    max_delay = 60

    while True:
        try:
            ws = await websockets.connect(url)
            retry_delay = 1  # reset on success
            return ws
        except Exception:
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)
```

---

## Robotics Networking

### Key Challenges in Robotics

1. **Low latency required** - control loops need <10ms response
2. **Unreliable wireless** - WiFi drops, interference
3. **Multiple devices** - sensors, actuators, compute units
4. **Real-time constraints** - can't wait for retransmission

### ROS 2 Networking (DDS)

ROS 2 uses DDS (Data Distribution Service) for communication:

```
┌────────────────────────────────────────────────────────┐
│ ROS 2 Network                                          │
│                                                        │
│   Node A              Node B              Node C       │
│  (sensor)           (planner)          (actuator)      │
│     │                  │                   │           │
│     │    /scan         │    /cmd_vel       │           │
│     └─────Topic────────┴─────Topic─────────┘           │
│                                                        │
│   Discovery: Automatic via multicast (no rosmaster!)   │
│   QoS: Configurable reliability, durability, deadline  │
└────────────────────────────────────────────────────────┘
```

**QoS (Quality of Service) settings:**
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For sensor data (accept some loss for low latency)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=1  # only latest value matters
)

# For commands (must be reliable)
command_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

### UDP for Real-Time Control

When TCP's reliability hurts more than helps:

```python
import socket
import struct

# Simple UDP telemetry sender
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_telemetry(x, y, theta, velocity):
    # Pack data efficiently
    data = struct.pack('!ffff', x, y, theta, velocity)
    sock.sendto(data, ('192.168.1.100', 9999))

# Receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 9999))
sock.setblocking(False)  # non-blocking for real-time

def receive_telemetry():
    try:
        data, addr = sock.recvfrom(16)
        return struct.unpack('!ffff', data)
    except BlockingIOError:
        return None  # no data available
```

### MQTT for IoT Sensors

Lightweight pub/sub protocol, great for sensor networks:

```python
import paho.mqtt.client as mqtt

# Publisher (sensor node)
client = mqtt.Client()
client.connect("mqtt-broker.local", 1883)
client.publish("robot/sensor/temperature", "23.5")

# Subscriber (central controller)
def on_message(client, userdata, msg):
    print(f"{msg.topic}: {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt-broker.local", 1883)
client.subscribe("robot/sensor/#")  # wildcard subscription
client.loop_forever()
```

**MQTT QoS levels:**
- QoS 0: At most once (fire and forget)
- QoS 1: At least once (may duplicate)
- QoS 2: Exactly once (highest overhead)

### Network Time Synchronization

Critical for sensor fusion and coordination:

```bash
# NTP - millisecond accuracy
sudo apt install ntp
ntpq -p  # check sync status

# PTP (Precision Time Protocol) - microsecond accuracy
sudo apt install linuxptp
sudo ptp4l -i eth0 -m

# Chrony - modern NTP, handles intermittent connectivity
sudo apt install chrony
chronyc sources
```

### Multicast for Discovery

Find devices without knowing their IPs:

```python
import socket
import struct

# Multicast sender
MCAST_GROUP = '224.1.1.1'
MCAST_PORT = 5007

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
sock.sendto(b"DISCOVER", (MCAST_GROUP, MCAST_PORT))

# Multicast receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', MCAST_PORT))

mreq = struct.pack("4sl", socket.inet_aton(MCAST_GROUP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

data, addr = sock.recvfrom(1024)
print(f"Received from {addr}: {data}")
```

---

## Mobile App Networking

### Handling Connectivity Changes

```swift
// iOS - Network framework
import Network

let monitor = NWPathMonitor()
monitor.pathUpdateHandler = { path in
    if path.status == .satisfied {
        if path.usesInterfaceType(.wifi) {
            print("WiFi connected - sync large data")
        } else if path.usesInterfaceType(.cellular) {
            print("Cellular - minimize data usage")
        }
    } else {
        print("Offline - use cached data")
    }
}
monitor.start(queue: DispatchQueue.global())
```

### Offline-First Architecture

```
┌─────────────────────────────────────────────────────┐
│ App                                                 │
│  ┌───────────────┐    ┌───────────────────────────┐│
│  │   UI Layer    │◄───│     Local Database        ││
│  └───────┬───────┘    │   (SQLite/Realm/Core Data)││
│          │            └───────────┬───────────────┘│
│          │                        │                │
│          ▼                        ▼                │
│  ┌───────────────┐    ┌───────────────────────────┐│
│  │   Sync Queue  │───▶│     Remote API            ││
│  │ (retry logic) │◄───│                           ││
│  └───────────────┘    └───────────────────────────┘│
└─────────────────────────────────────────────────────┘

1. Write to local DB first (instant UI response)
2. Queue changes for sync
3. Sync when online
4. Handle conflicts (last-write-wins, merge, etc.)
```

### Certificate Pinning

Prevent man-in-the-middle attacks:

```swift
// iOS - URLSession with pinning
class PinningDelegate: NSObject, URLSessionDelegate {
    func urlSession(_ session: URLSession,
                    didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {

        guard let serverTrust = challenge.protectionSpace.serverTrust,
              let certificate = SecTrustGetCertificateAtIndex(serverTrust, 0) else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        let serverCertData = SecCertificateCopyData(certificate) as Data
        let pinnedCertData = // load your pinned cert

        if serverCertData == pinnedCertData {
            completionHandler(.useCredential, URLCredential(trust: serverTrust))
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }
}
```

### Background Tasks

```swift
// iOS - Background URL Session
let config = URLSessionConfiguration.background(withIdentifier: "com.app.sync")
config.isDiscretionary = true  // iOS chooses optimal time
config.sessionSendsLaunchEvents = true  // wake app on completion

let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
let task = session.downloadTask(with: url)
task.resume()  // continues even if app is killed
```

---

## Network Security Basics

### Authentication Patterns

| Pattern | When to Use |
|---------|-------------|
| API Key | Simple server-to-server |
| Basic Auth | Internal tools (always over HTTPS!) |
| Bearer Token (JWT) | User authentication, stateless |
| OAuth 2.0 | Third-party integration |
| mTLS | High-security machine-to-machine |

### JWT (JSON Web Tokens)

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.  ← Header (base64)
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.  ← Payload (base64)
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c  ← Signature

Decoded payload: {"sub": "1234567890", "name": "John", "exp": 1609459200}
```

**Important:** JWTs are signed, not encrypted! Anyone can read the payload.

### Common Vulnerabilities

**SQL Injection:**
```python
# BAD - never do this
query = f"SELECT * FROM users WHERE id = {user_input}"

# GOOD - parameterized queries
cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))
```

**SSRF (Server-Side Request Forgery):**
```python
# BAD - user controls URL
response = requests.get(user_provided_url)

# GOOD - validate URL
from urllib.parse import urlparse
parsed = urlparse(user_provided_url)
if parsed.hostname not in ALLOWED_HOSTS:
    raise ValueError("Invalid host")
```

---

## Performance Optimization

### Reducing Latency

| Technique | Impact |
|-----------|--------|
| Connection pooling | Avoid TCP handshake overhead |
| HTTP/2 | Multiplexing, header compression |
| Compression (gzip) | Less data to transfer |
| CDN | Content closer to users |
| DNS prefetching | Resolve domains early |
| Keep-Alive | Reuse connections |

### Reducing Bandwidth

```python
# Pagination - don't fetch everything
GET /items?page=1&limit=20

# Sparse fieldsets - only fetch needed fields
GET /users?fields=id,name

# Conditional requests - skip if unchanged
GET /resource
If-None-Match: "etag-value"
→ 304 Not Modified (no body)

# Compression
Accept-Encoding: gzip, deflate
```

### Batching Requests

```python
# BAD - N+1 requests
for user_id in user_ids:
    user = api.get_user(user_id)

# GOOD - batch request
users = api.get_users(ids=user_ids)  # single request
```

---

## Common Gotchas

### 1. DNS Caching

DNS results are cached. When you change a server's IP:
```
- Browser cache: minutes
- OS cache: minutes to hours
- ISP cache: hours
- Your code's cache: forever (if not handled)
```

**Fix:** Set appropriate TTLs, use connection managers that respect TTL.

### 2. TCP TIME_WAIT

After closing a connection, the port stays in TIME_WAIT for ~60s:
```bash
netstat -an | grep TIME_WAIT
```

**Fix:** Reuse connections (connection pooling), or set `SO_REUSEADDR`.

### 3. MTU Issues

Maximum Transmission Unit - largest packet size (usually 1500 bytes).

**Symptoms:** Large packets fail, small ones work.

**Debug:**
```bash
ping -s 1472 google.com  # 1472 + 28 header = 1500
```

### 4. Half-Open Connections

One side thinks connection is open, other side closed it.

**Fix:** Implement heartbeats/keepalives:
```python
sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
```

### 5. Blocking I/O in Event Loops

```python
# BAD - blocks the entire event loop
async def bad_handler():
    response = requests.get(url)  # blocking!

# GOOD - use async HTTP client
async def good_handler():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### 6. Localhost vs 0.0.0.0

```python
# Only accessible from same machine
server.bind(('127.0.0.1', 8080))
# or
server.bind(('localhost', 8080))

# Accessible from any network interface
server.bind(('0.0.0.0', 8080))
```

### 7. IPv6 Surprises

Your code might unexpectedly use IPv6:
```python
# Force IPv4
socket.create_connection((host, port), source_address=('0.0.0.0', 0))

# Handle both
socket.setdefaulttimeout(5)  # prevent hanging on unreachable IPv6
```

---

## Quick Reference

### Must-Know Ports
```
22   SSH
53   DNS
80   HTTP
443  HTTPS
1883 MQTT
5432 PostgreSQL
6379 Redis
8080 HTTP Alt
9090 Prometheus
```

### Debug Commands
```bash
ping host              # connectivity
traceroute host        # path
dig domain             # DNS
curl -v url            # HTTP debug
nc -zv host port       # port check
ss -tulpn              # listening ports
tcpdump -i any port X  # packet capture
```

### Status Code Quick Check
```
2xx = Good
3xx = Redirect
4xx = Your fault
5xx = Their fault

200 = OK
201 = Created
204 = OK (no body)
400 = Bad request
401 = Login required
403 = Not allowed
404 = Not found
429 = Slow down
500 = Server broke
503 = Try later
```
