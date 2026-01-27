# Networking Fundamentals

Everything you need to know about how the internet works, without the textbook fluff.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [The Layer Model](#the-layer-model)
3. [IP Addresses & Subnets](#ip-addresses--subnets)
4. [DNS - The Internet's Phone Book](#dns---the-internets-phone-book)
5. [TCP vs UDP](#tcp-vs-udp)
6. [Ports & Sockets](#ports--sockets)
7. [HTTP & HTTPS](#http--https)
8. [How Data Actually Travels](#how-data-actually-travels)
9. [NAT & Firewalls](#nat--firewalls)
10. [Common Protocols Cheat Sheet](#common-protocols-cheat-sheet)

---

## The Big Picture

The internet is just **computers talking to each other** using agreed-upon rules (protocols).

```
Your Computer ←→ Router ←→ ISP ←→ Internet Backbone ←→ ISP ←→ Server
     │                              │
     └── Your local network         └── Global network of networks
```

**Key insight**: The internet is decentralized. There's no single "internet computer" - it's millions of interconnected networks.

---

## The Layer Model

Networks are organized in **layers**. Each layer only talks to the layer above/below it.

### The Practical Model (TCP/IP - 4 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  What you interact with: HTTP, HTTPS, DNS, SSH, FTP, SMTP   │
│  "I want to load google.com"                                │
├─────────────────────────────────────────────────────────────┤
│  TRANSPORT LAYER                                            │
│  How data is delivered: TCP (reliable) or UDP (fast)        │
│  "Break into packets, ensure delivery, reassemble"          │
├─────────────────────────────────────────────────────────────┤
│  NETWORK/INTERNET LAYER                                     │
│  Addressing & routing: IP (IPv4, IPv6)                      │
│  "Route this to IP address 142.250.80.46"                   │
├─────────────────────────────────────────────────────────────┤
│  LINK/NETWORK ACCESS LAYER                                  │
│  Physical transmission: Ethernet, WiFi, 5G                  │
│  "Send these electrical signals over the wire"              │
└─────────────────────────────────────────────────────────────┘
```

### Why layers matter

- **Abstraction**: Your app doesn't care if you're on WiFi or Ethernet
- **Modularity**: Can upgrade one layer without touching others
- **Debugging**: Problems can be isolated to specific layers

---

## IP Addresses & Subnets

### IPv4 Addresses

An IP address is like a phone number for your device.

```
192.168.1.100
 │   │   │  │
 └───┴───┴──┴── Four numbers (0-255), called "octets"

Total: 4 bytes = 32 bits = ~4.3 billion possible addresses
```

### Special IP Ranges

| Range | Purpose |
|-------|---------|
| `127.0.0.1` | Localhost (yourself) |
| `10.0.0.0/8` | Private network (large orgs) |
| `172.16.0.0/12` | Private network (medium) |
| `192.168.0.0/16` | Private network (home/small office) |
| `0.0.0.0` | "Any" address (listen on all interfaces) |
| `255.255.255.255` | Broadcast address |

### Subnets & CIDR Notation

A subnet divides a network into smaller chunks.

```
192.168.1.0/24
           └── "24 bits are the network part"

Means: 
- Network: 192.168.1.x
- Usable hosts: 192.168.1.1 - 192.168.1.254 (254 devices)
- Broadcast: 192.168.1.255
```

**Common subnet masks:**
- `/8` = 255.0.0.0 → 16 million hosts
- `/16` = 255.255.0.0 → 65,534 hosts
- `/24` = 255.255.255.0 → 254 hosts
- `/32` = single host

### IPv6

IPv4 addresses ran out. IPv6 fixes this with 128-bit addresses:

```
2001:0db8:85a3:0000:0000:8a2e:0370:7334

Shorthand: 2001:db8:85a3::8a2e:370:7334
           (leading zeros and consecutive zeros can be omitted)
```

---

## DNS - The Internet's Phone Book

DNS translates human-readable names to IP addresses.

```
google.com → 142.250.80.46
```

### How DNS Resolution Works

```
1. You type "google.com" in browser

2. Browser checks cache → not found

3. OS checks /etc/hosts → not found

4. Query local DNS resolver (usually your router or ISP)
   └→ Resolver checks its cache → not found

5. Resolver queries Root DNS server
   └→ "I don't know google.com, but .com is handled by these servers"

6. Resolver queries .com TLD server
   └→ "google.com is handled by ns1.google.com"

7. Resolver queries Google's nameserver
   └→ "google.com is 142.250.80.46"

8. Answer cached at each level, returned to browser
```

### DNS Record Types

| Type | Purpose | Example |
|------|---------|---------|
| **A** | Domain → IPv4 | `google.com → 142.250.80.46` |
| **AAAA** | Domain → IPv6 | `google.com → 2607:f8b0:4004:800::200e` |
| **CNAME** | Alias to another domain | `www.example.com → example.com` |
| **MX** | Mail server | `gmail.com → alt1.gmail-smtp-in.l.google.com` |
| **TXT** | Arbitrary text | Used for verification, SPF, DKIM |
| **NS** | Nameserver | `google.com → ns1.google.com` |

### Useful DNS Commands

```bash
# Look up IP address
nslookup google.com
dig google.com

# See full DNS resolution path
dig +trace google.com

# Query specific record type
dig google.com MX
dig google.com TXT
```

---

## TCP vs UDP

The two main transport protocols. **This is important!**

### TCP (Transmission Control Protocol)

**Reliable, ordered, connection-based**

```
┌──────────────────────────────────────────────────────────┐
│ TCP Three-Way Handshake (Connection Setup)               │
│                                                          │
│ Client                              Server               │
│   │                                   │                  │
│   │─────── SYN (seq=100) ────────────▶│                  │
│   │                                   │                  │
│   │◀────── SYN-ACK (seq=300,ack=101)──│                  │
│   │                                   │                  │
│   │─────── ACK (ack=301) ────────────▶│                  │
│   │                                   │                  │
│   │         Connection established!   │                  │
└──────────────────────────────────────────────────────────┘
```

**Features:**
- ✅ Guaranteed delivery (retransmits lost packets)
- ✅ Ordered (packets reassembled in correct order)
- ✅ Flow control (doesn't overwhelm receiver)
- ✅ Congestion control (adapts to network conditions)
- ❌ More overhead (headers, handshakes, ACKs)
- ❌ Higher latency

**Use for:** Web, email, file transfer, APIs - anything where data must be correct

### UDP (User Datagram Protocol)

**Fast, no guarantees, connectionless**

```
Client                              Server
  │                                   │
  │─────── Data packet ──────────────▶│   (no handshake)
  │─────── Data packet ──────────────▶│   (no confirmation)
  │─────── Data packet ────────X      │   (packet lost? oh well)
  │                                   │
```

**Features:**
- ✅ Very fast (no handshake, no waiting for ACKs)
- ✅ Low overhead (small header)
- ✅ Supports broadcast/multicast
- ❌ No delivery guarantee
- ❌ No ordering
- ❌ Packets can be duplicated

**Use for:** Video streaming, gaming, VoIP, DNS, IoT sensors - when speed matters more than perfection

### Quick Comparison

| Aspect | TCP | UDP |
|--------|-----|-----|
| Connection | Required (handshake) | None |
| Reliability | Guaranteed | Best-effort |
| Ordering | Guaranteed | None |
| Speed | Slower | Faster |
| Header size | 20+ bytes | 8 bytes |
| Use case | Web, files, APIs | Streaming, gaming, IoT |

---

## Ports & Sockets

### Ports

An IP address gets you to a computer. A **port** gets you to a specific application.

```
IP Address = Street address of a building
Port       = Apartment number

192.168.1.100:8080
     │          │
     IP      Port
```

**Port ranges:**
- `0-1023`: Well-known ports (require root/admin)
- `1024-49151`: Registered ports
- `49152-65535`: Dynamic/private ports

**Common ports:**

| Port | Service |
|------|---------|
| 20, 21 | FTP |
| 22 | SSH |
| 23 | Telnet (insecure!) |
| 25 | SMTP (email) |
| 53 | DNS |
| 80 | HTTP |
| 443 | HTTPS |
| 3306 | MySQL |
| 5432 | PostgreSQL |
| 6379 | Redis |
| 8080 | HTTP alternate |
| 27017 | MongoDB |

### Sockets

A **socket** is an endpoint for communication: `IP:Port`

```
Socket = (IP Address, Port, Protocol)

Example: (192.168.1.100, 8080, TCP)
```

When you open a connection, you create a socket pair:
```
Your socket:   192.168.1.100:54321  (random high port)
Server socket: 142.250.80.46:443   (HTTPS)
```

---

## HTTP & HTTPS

### HTTP Request Structure

```
GET /api/users HTTP/1.1          ← Request line (method, path, version)
Host: api.example.com            ← Headers start
Authorization: Bearer xyz123
Content-Type: application/json
Accept: application/json
                                 ← Empty line separates headers from body
{"name": "John"}                 ← Body (optional)
```

### HTTP Methods

| Method | Purpose | Idempotent? | Has Body? |
|--------|---------|-------------|-----------|
| GET | Retrieve data | Yes | No |
| POST | Create resource | No | Yes |
| PUT | Replace resource | Yes | Yes |
| PATCH | Partial update | No | Yes |
| DELETE | Remove resource | Yes | Usually no |
| HEAD | GET without body | Yes | No |
| OPTIONS | Get allowed methods | Yes | No |

### HTTP Status Codes

```
1xx - Informational (rarely seen)
2xx - Success
3xx - Redirection
4xx - Client error (your fault)
5xx - Server error (their fault)
```

**Must-know codes:**

| Code | Meaning |
|------|---------|
| 200 | OK |
| 201 | Created |
| 204 | No Content (success, empty response) |
| 301 | Moved Permanently |
| 302 | Found (temporary redirect) |
| 304 | Not Modified (use cache) |
| 400 | Bad Request |
| 401 | Unauthorized (need to login) |
| 403 | Forbidden (logged in but not allowed) |
| 404 | Not Found |
| 405 | Method Not Allowed |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 502 | Bad Gateway |
| 503 | Service Unavailable |
| 504 | Gateway Timeout |

### HTTPS

HTTPS = HTTP + TLS (Transport Layer Security)

```
1. TCP handshake (SYN, SYN-ACK, ACK)

2. TLS handshake:
   - Client: "Hello, I support these encryption methods"
   - Server: "Let's use this one, here's my certificate"
   - Client: Verifies certificate against trusted CAs
   - Both: Exchange keys using asymmetric crypto
   - Both: Switch to symmetric encryption (faster)

3. Encrypted HTTP communication
```

**What TLS provides:**
- 🔒 **Encryption**: Data can't be read in transit
- 🔐 **Authentication**: Server proves its identity
- ✅ **Integrity**: Data can't be modified in transit

---

## How Data Actually Travels

Let's trace a request from your browser to Google:

```
1. APPLICATION: Browser creates HTTP request
   GET / HTTP/1.1
   Host: google.com

2. TRANSPORT: TCP wraps it in a segment
   ┌─────────────────────────────────────┐
   │ TCP Header │ HTTP Request           │
   │ (src port, dst port, seq#, flags)   │
   └─────────────────────────────────────┘

3. NETWORK: IP wraps it in a packet
   ┌─────────────────────────────────────┐
   │ IP Header │ TCP Segment             │
   │ (src IP, dst IP, TTL)               │
   └─────────────────────────────────────┘

4. LINK: Ethernet wraps it in a frame
   ┌─────────────────────────────────────┐
   │ Eth Header │ IP Packet │ Eth Footer │
   │ (MAC addrs)│           │ (checksum) │
   └─────────────────────────────────────┘

5. Physical: Converted to electrical/optical signals
```

### Packet Journey Through the Network

```
Your Computer
     │
     ▼ (check: is destination on local network?)
     │
     │ No → send to default gateway (router)
     ▼
Your Router (NAT)
     │
     │ Replaces your private IP with public IP
     ▼
ISP Router
     │
     ▼
Internet Backbone (multiple hops)
     │
     │ Each router looks at destination IP
     │ and forwards to next hop
     ▼
Google's Network
     │
     ▼
Google's Load Balancer
     │
     ▼
Google's Server
```

You can see this with `traceroute`:
```bash
traceroute google.com
```

---

## NAT & Firewalls

### NAT (Network Address Translation)

Your home network uses private IPs (192.168.x.x), but the internet needs public IPs. NAT translates between them.

```
┌─────────────────────────────────────────────────────────┐
│ Your Home Network                                       │
│                                                         │
│  Phone        Laptop       Desktop                      │
│  192.168.1.10 192.168.1.11 192.168.1.12                │
│       │            │            │                       │
│       └────────────┼────────────┘                       │
│                    ▼                                    │
│              ┌──────────┐                               │
│              │  Router  │ ← NAT happens here            │
│              │ (NAT)    │                               │
│              └────┬─────┘                               │
│                   │                                     │
└───────────────────┼─────────────────────────────────────┘
                    │ Public IP: 73.45.123.99
                    ▼
               Internet
```

**NAT table example:**
```
Internal          External (what the internet sees)
192.168.1.10:54321 → 73.45.123.99:30001
192.168.1.11:54322 → 73.45.123.99:30002
```

**NAT implications:**
- Devices behind NAT can't be directly reached from internet
- Need port forwarding or NAT traversal for incoming connections
- This is why peer-to-peer is tricky (both sides behind NAT)

### Firewalls

Firewalls filter traffic based on rules.

```
Common rules:
- Allow outgoing connections (you can browse the web)
- Block incoming connections (others can't connect to you)
- Allow established connections (responses to your requests)
- Allow specific ports (SSH on 22, HTTP on 80)
```

**Firewall commands (Linux):**
```bash
# View rules
sudo iptables -L

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Modern Linux uses ufw (simpler)
sudo ufw allow 22
sudo ufw enable
```

---

## Common Protocols Cheat Sheet

### Application Layer

| Protocol | Port | Purpose |
|----------|------|---------|
| HTTP | 80 | Web (unencrypted) |
| HTTPS | 443 | Web (encrypted) |
| DNS | 53 | Domain name resolution |
| SSH | 22 | Secure shell/tunneling |
| FTP | 21 | File transfer |
| SMTP | 25 | Send email |
| IMAP | 143/993 | Receive email |
| MQTT | 1883/8883 | IoT messaging |
| WebSocket | 80/443 | Full-duplex communication |
| gRPC | varies | RPC framework |

### Network Layer

| Protocol | Purpose |
|----------|---------|
| IP | Addressing and routing |
| ICMP | Error messages, ping |
| ARP | IP → MAC address resolution |

### Useful Commands

```bash
# Check connectivity
ping google.com

# See your IP
ip addr                    # Linux
ifconfig                   # macOS

# See open ports
netstat -tulpn             # Linux
lsof -i -P                 # macOS

# See routing table
ip route                   # Linux
netstat -rn                # macOS

# Capture packets
sudo tcpdump -i any port 80

# Test if port is open
nc -zv google.com 443
telnet google.com 443

# See active connections
ss -tuln                   # Linux
netstat -an                # All platforms
```

---

## Key Takeaways

1. **Layers abstract complexity** - your app talks HTTP, doesn't care about Ethernet frames
2. **TCP = reliable + slow, UDP = fast + unreliable** - choose based on your needs
3. **NAT lets many devices share one public IP** - but makes incoming connections hard
4. **DNS translates names to IPs** - and is often the source of "it's not working" issues
5. **Ports let multiple services run on one IP** - memorize the common ones
6. **HTTPS encrypts everything** - always use it in production

---

Next: [Practical Networking](./02_PRACTICAL.md) - Tools and techniques for robotics and app development
