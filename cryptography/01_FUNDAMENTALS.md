# Cryptography Fundamentals

The essential concepts behind securing data, without the math-heavy proofs.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Symmetric Encryption](#symmetric-encryption)
3. [Asymmetric Encryption](#asymmetric-encryption)
4. [Hash Functions](#hash-functions)
5. [Digital Signatures](#digital-signatures)
6. [Key Exchange](#key-exchange)
7. [Certificates & PKI](#certificates--pki)
8. [How TLS/HTTPS Actually Works](#how-tlshttps-actually-works)
9. [Random Numbers](#random-numbers)
10. [Algorithm Cheat Sheet](#algorithm-cheat-sheet)

---

## The Big Picture

Cryptography solves four fundamental problems:

| Problem | Solution | Example |
|---------|----------|---------|
| **Confidentiality** | Encryption | Only intended recipient can read |
| **Integrity** | Hashing/MAC | Data hasn't been tampered with |
| **Authentication** | Signatures/Certificates | Sender is who they claim to be |
| **Non-repudiation** | Digital Signatures | Sender can't deny sending |

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRYPTOGRAPHY LANDSCAPE                        │
│                                                                  │
│  Symmetric Crypto              Asymmetric Crypto                │
│  ────────────────              ─────────────────                │
│  Same key to encrypt           Different keys:                  │
│  and decrypt                   Public (share) + Private (keep)  │
│                                                                  │
│  ✓ Very fast                   ✓ Solves key distribution        │
│  ✗ Key distribution problem    ✗ Much slower (100-1000x)        │
│                                                                  │
│  Uses: Bulk data encryption    Uses: Key exchange, signatures   │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Hash Functions                Digital Signatures               │
│  ──────────────                ──────────────────               │
│  One-way: can't reverse        Prove authenticity               │
│  Fixed output size             Only owner of private key        │
│                                can create                        │
│                                                                  │
│  Uses: Integrity, passwords    Uses: Auth, non-repudiation      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Symmetric Encryption

**Same key** for encryption and decryption.

```
Plaintext: "Hello World"
    │
    ▼
┌─────────────────────┐
│   Encrypt with      │ ◀── Key: "mysecretkey123"
│   AES-256           │
└─────────┬───────────┘
          │
          ▼
Ciphertext: "3J8fK2mNpQ..."
          │
          ▼
┌─────────────────────┐
│   Decrypt with      │ ◀── Same Key: "mysecretkey123"
│   AES-256           │
└─────────┬───────────┘
          │
          ▼
Plaintext: "Hello World"
```

### Block Ciphers vs Stream Ciphers

**Block Cipher:** Encrypts fixed-size blocks (e.g., 128 bits)
- AES, DES, Blowfish
- Need a "mode of operation" for data larger than one block

**Stream Cipher:** Encrypts one bit/byte at a time
- ChaCha20, RC4 (deprecated)
- Good for streaming data

### Block Cipher Modes

How to handle data larger than one block:

| Mode | Description | Use Case |
|------|-------------|----------|
| **ECB** | Each block encrypted independently | ❌ Never use (patterns leak) |
| **CBC** | Each block XORed with previous | ⚠️ Legacy, needs padding |
| **CTR** | Block cipher → stream cipher | ✓ Parallelizable |
| **GCM** | CTR + authentication | ✓ **Recommended** |

**Why ECB is dangerous:**

```
Original image          ECB encrypted          CBC/GCM encrypted
┌────────────┐         ┌────────────┐         ┌────────────┐
│ ██████████ │         │ ████  ████ │         │ ░▒▓█░▒▓█░▒ │
│ ██      ██ │   →     │ ████  ████ │         │ ▓█░▒▓█░▒▓█ │
│ ██████████ │         │ ██████████ │         │ ░▒▓█░▒▓█░▒ │
└────────────┘         └────────────┘         └────────────┘
                       Pattern preserved!     Looks random
```

### AES (Advanced Encryption Standard)

The gold standard for symmetric encryption:

- **AES-128**: 128-bit key, very secure
- **AES-192**: 192-bit key
- **AES-256**: 256-bit key, used for top secret

**Always use AES-GCM** (Galois/Counter Mode):
- Provides encryption + authentication
- Detects if ciphertext was tampered
- Fast (hardware acceleration on most CPUs)

### The IV/Nonce

**Never encrypt twice with the same key + IV combination!**

```
Plaintext + Key + IV → Ciphertext

The IV (Initialization Vector) or Nonce:
- Makes same plaintext produce different ciphertext
- Must be unique for each encryption with same key
- Usually 12-16 bytes of random data
- Can be sent alongside ciphertext (not secret)
```

---

## Asymmetric Encryption

**Two keys**: Public key (share freely) and Private key (keep secret).

```
┌─────────────────────────────────────────────────────────────┐
│ Key Pair Generation                                         │
│                                                             │
│     ┌───────────────┐                                       │
│     │   Generate    │                                       │
│     │   Key Pair    │                                       │
│     └───────┬───────┘                                       │
│             │                                               │
│     ┌───────┴───────┐                                       │
│     ▼               ▼                                       │
│ ┌─────────┐   ┌─────────────┐                               │
│ │ Public  │   │   Private   │                               │
│ │  Key    │   │    Key      │                               │
│ └────┬────┘   └──────┬──────┘                               │
│      │               │                                      │
│      ▼               ▼                                      │
│  Share with      Keep secret!                               │
│   everyone       Never share                                │
└─────────────────────────────────────────────────────────────┘
```

### Two Uses of Asymmetric Crypto

**1. Encryption (Confidentiality):**
```
Encrypt with PUBLIC key  → Only PRIVATE key can decrypt

Alice wants to send secret to Bob:
1. Alice encrypts with Bob's PUBLIC key
2. Only Bob can decrypt with his PRIVATE key
```

**2. Signing (Authentication):**
```
Sign with PRIVATE key → Anyone can verify with PUBLIC key

Bob wants to prove he sent a message:
1. Bob signs with his PRIVATE key
2. Anyone can verify with Bob's PUBLIC key
```

### RSA

The classic asymmetric algorithm (1977):

- Based on difficulty of factoring large primes
- Key sizes: 2048-bit minimum, 4096-bit recommended
- **Slow** - only used for small data or key exchange
- Being phased out in favor of elliptic curves

### Elliptic Curve Cryptography (ECC)

Modern asymmetric crypto:

- Much smaller keys for same security
- Faster than RSA
- Used in TLS, Bitcoin, Signal, etc.

**Key size comparison (equivalent security):**

| RSA | ECC | Security Level |
|-----|-----|----------------|
| 2048-bit | 256-bit | 112-bit |
| 3072-bit | 384-bit | 128-bit |
| 4096-bit | 512-bit | 140-bit |

**Common curves:**
- **P-256 (secp256r1)**: NIST standard, widely used
- **P-384 (secp384r1)**: Higher security
- **Curve25519**: Modern, designed for security (used in Signal)
- **secp256k1**: Used in Bitcoin

---

## Hash Functions

**One-way function**: Easy to compute, impossible to reverse.

```
Input (any size)              Output (fixed size)
─────────────────             ───────────────────

"Hello"           ──SHA-256──▶ 185f8db32271fe25f561...
"Hello "          ──SHA-256──▶ 7f83b1657ff1fc53b92d...  (completely different!)
<1GB file>        ──SHA-256──▶ 9f86d081884c7d659a2f...  (still 256 bits)
```

### Properties of Cryptographic Hashes

1. **Deterministic**: Same input → same output, always
2. **Fast**: Quick to compute
3. **One-way**: Can't get input from output
4. **Collision-resistant**: Hard to find two inputs with same output
5. **Avalanche effect**: Small input change → completely different output

### Common Hash Functions

| Algorithm | Output Size | Status |
|-----------|-------------|--------|
| MD5 | 128-bit | ❌ Broken (collisions found) |
| SHA-1 | 160-bit | ❌ Broken (don't use for security) |
| SHA-256 | 256-bit | ✓ Secure |
| SHA-384 | 384-bit | ✓ Secure |
| SHA-512 | 512-bit | ✓ Secure |
| SHA-3 | Variable | ✓ Newest standard |
| BLAKE2 | Variable | ✓ Fast and secure |
| BLAKE3 | 256-bit | ✓ Very fast |

### Hash Use Cases

**1. Data Integrity:**
```
Download file + published hash
Compute hash of downloaded file
Compare → if match, file wasn't corrupted/tampered
```

**2. Password Storage** (with special password hashes):
```
Store: hash(password + salt)
Verify: hash(input + salt) == stored_hash
```

**3. Commit IDs:**
```
Git commit: SHA-1 hash of content + metadata
```

**4. Deduplication:**
```
Same hash = same content (with high probability)
```

### HMAC (Hash-based Message Authentication Code)

Hash + secret key = proof of authenticity + integrity

```
HMAC(key, message) → authentication tag

Only someone with the key can:
1. Create a valid tag
2. Verify a tag

Used in: API authentication, cookie signing, JWT
```

---

## Digital Signatures

**Prove a message came from you** and wasn't modified.

```
┌─────────────────────────────────────────────────────────────┐
│ Signing (by sender)                                         │
│                                                             │
│   Message: "Transfer $100 to Alice"                         │
│                │                                            │
│                ▼                                            │
│   ┌───────────────────────┐                                 │
│   │   Hash the message    │                                 │
│   └───────────┬───────────┘                                 │
│               │                                             │
│               ▼                                             │
│   ┌───────────────────────┐                                 │
│   │  Encrypt hash with    │ ◀── Private Key                 │
│   │  PRIVATE key          │                                 │
│   └───────────┬───────────┘                                 │
│               │                                             │
│               ▼                                             │
│         Signature                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Verification (by anyone)                                    │
│                                                             │
│   Received: Message + Signature                             │
│                │          │                                 │
│                ▼          ▼                                 │
│   ┌──────────────┐  ┌──────────────────┐                   │
│   │ Hash message │  │ Decrypt sig with │◀── Public Key     │
│   └──────┬───────┘  │ PUBLIC key       │                   │
│          │          └────────┬─────────┘                    │
│          │                   │                              │
│          ▼                   ▼                              │
│       Hash A              Hash B                            │
│          │                   │                              │
│          └─────────┬─────────┘                              │
│                    ▼                                        │
│             Compare hashes                                  │
│         Match? → Valid signature!                           │
└─────────────────────────────────────────────────────────────┘
```

### Signature Algorithms

| Algorithm | Description |
|-----------|-------------|
| **RSA-PSS** | RSA-based, widely supported |
| **ECDSA** | Elliptic curve based, smaller signatures |
| **EdDSA (Ed25519)** | Modern, fast, recommended |

### Why Hash Then Sign?

1. **Speed**: Signing is slow; hash is fast + fixed size
2. **Signature size**: Signature of hash = signature of any size message
3. **Security**: Some signature schemes need fixed-size input

---

## Key Exchange

**Problem**: How do two parties agree on a secret key over an insecure channel?

### Diffie-Hellman Key Exchange

```
Alice                                              Bob
──────                                            ─────
Has: private value 'a'                    Has: private value 'b'
     public params (g, p)                      public params (g, p)

Computes: A = g^a mod p    ──────────▶    Receives A
          (public value)

Receives B                 ◀──────────    Computes: B = g^b mod p
                                                    (public value)

Computes: s = B^a mod p                   Computes: s = A^b mod p
             = g^(ab) mod p                          = g^(ab) mod p

     Shared Secret: s                          Shared Secret: s
        (identical!)                              (identical!)
```

**The magic**: 
- Anyone can see A, B, g, p
- But computing 'ab' from just A, B is computationally infeasible

### ECDH (Elliptic Curve Diffie-Hellman)

Same concept, but using elliptic curves:
- Smaller values, same security
- Faster computation
- Standard in modern TLS

---

## Certificates & PKI

**Problem**: How do you know a public key really belongs to who they claim?

### The Certificate

A certificate binds an identity to a public key, signed by a trusted authority.

```
┌─────────────────────────────────────────────────────────────┐
│ X.509 Certificate                                           │
│                                                             │
│   Subject: CN=www.google.com                                │
│   Issuer: CN=GTS CA 1C3 (Certificate Authority)             │
│   Valid: Jan 2024 - Apr 2024                                │
│   Public Key: [Google's public key]                         │
│   ...                                                       │
│                                                             │
│   ─────────────────────────────────────────────────────     │
│   Signature: [Signed by GTS CA 1C3's private key]           │
│   (Anyone can verify using CA's public key)                 │
└─────────────────────────────────────────────────────────────┘
```

### Chain of Trust

```
Root CA (self-signed, pre-installed in OS/browser)
    │
    │ signs
    ▼
Intermediate CA
    │
    │ signs
    ▼
End-Entity Certificate (website's cert)
    │
    │ contains
    ▼
Website's Public Key
```

**Your browser/OS ships with ~100-150 trusted root CAs.**

### Certificate Validation

When you visit https://example.com:

1. Server sends its certificate (+ intermediate certs)
2. Browser checks:
   - Is cert signed by trusted CA (or chain to trusted CA)?
   - Is cert not expired?
   - Does cert match the domain (example.com)?
   - Is cert not revoked (CRL/OCSP check)?
3. If all pass → connection is secure

### Certificate Transparency

Public logs of all issued certificates:
- Anyone can monitor for suspicious certs
- Browsers require certificates to be logged
- You can check: https://crt.sh

---

## How TLS/HTTPS Actually Works

TLS combines everything we've learned:

```
┌─────────────────────────────────────────────────────────────┐
│ TLS 1.3 Handshake (simplified)                              │
│                                                             │
│ Client                                      Server          │
│ ──────                                      ──────          │
│                                                             │
│ ClientHello ─────────────────────────────────▶              │
│   • Supported cipher suites                                 │
│   • Random value                                            │
│   • Key share (ECDH public value)                           │
│                                                             │
│            ◀───────────────────────────── ServerHello       │
│                                  • Chosen cipher suite      │
│                                  • Random value             │
│                                  • Key share (ECDH public)  │
│                                  • Certificate              │
│                                  • Certificate Verify (sig) │
│                                  • Finished                 │
│                                                             │
│   [Both compute shared secret from ECDH]                    │
│   [Derive encryption keys from shared secret]               │
│                                                             │
│ Finished ─────────────────────────────────▶                 │
│   (encrypted with derived keys)                             │
│                                                             │
│ ═══════════════════════════════════════════════════════════ │
│       All further communication encrypted with AES-GCM      │
└─────────────────────────────────────────────────────────────┘
```

### What TLS Provides

| Property | How |
|----------|-----|
| **Confidentiality** | AES-GCM encryption |
| **Integrity** | GCM authentication tag |
| **Server Authentication** | Certificate + signature |
| **Forward Secrecy** | Ephemeral ECDH keys |

### Forward Secrecy

Even if server's private key is stolen later, past conversations remain secure.

```
Old TLS (RSA key exchange):
- Client encrypts premaster secret with server's RSA public key
- If attacker records traffic AND later gets private key → can decrypt all

TLS 1.3 (ECDHE):
- Fresh ECDH keys generated for each session
- Session key never touches long-term private key
- Attacker with private key can't decrypt past sessions
```

---

## Random Numbers

**Randomness is critical.** Bad randomness has broken real-world crypto.

### CSPRNG (Cryptographically Secure PRNG)

| Source | When to Use |
|--------|-------------|
| `/dev/urandom` | Linux/Mac - always available, secure |
| `/dev/random` | Linux - blocks if low entropy (rarely needed) |
| `CryptGenRandom` | Windows |
| Hardware RNG | Available on modern CPUs (RDRAND) |

**Never use:**
- `rand()` in C
- `Math.random()` in JavaScript
- `random.random()` in Python (use `secrets` module instead)

### Common Mistakes

```
❌ Using predictable seed (time, PID)
❌ Using non-crypto random for keys/IVs
❌ Reusing nonces
❌ Weak entropy at boot time (VMs, embedded)
```

---

## Algorithm Cheat Sheet

### What to Use in 2024+

| Purpose | Recommended | Avoid |
|---------|-------------|-------|
| **Symmetric Encryption** | AES-256-GCM, ChaCha20-Poly1305 | DES, 3DES, AES-ECB, RC4 |
| **Hashing** | SHA-256, SHA-3, BLAKE2/3 | MD5, SHA-1 |
| **Password Hashing** | Argon2id, bcrypt, scrypt | MD5, SHA-*, plain hash |
| **Key Exchange** | ECDH (Curve25519, P-256) | RSA < 2048, DH < 2048 |
| **Signatures** | Ed25519, ECDSA (P-256), RSA-PSS | RSA-PKCS1v1.5 for new code |
| **MAC** | HMAC-SHA256, Poly1305 | HMAC-MD5, HMAC-SHA1 |

### Key Sizes

| Algorithm | Minimum | Recommended |
|-----------|---------|-------------|
| AES | 128-bit | 256-bit |
| RSA | 2048-bit | 3072+ bit |
| ECDSA/ECDH | 256-bit (P-256) | 256-bit |
| Ed25519 | 256-bit | 256-bit (fixed) |

### Security Levels

```
80-bit   → Breakable with resources (deprecated)
112-bit  → Minimum acceptable
128-bit  → Standard security
192-bit  → High security
256-bit  → Post-quantum resistant (symmetric)
```

---

## Quick Reference

### Symmetric vs Asymmetric

| Aspect | Symmetric | Asymmetric |
|--------|-----------|------------|
| Keys | One shared key | Public + Private pair |
| Speed | Fast (100-1000x) | Slow |
| Key length | 128-256 bits | 2048-4096 bits (RSA) |
| Use case | Bulk encryption | Key exchange, signatures |
| Examples | AES, ChaCha20 | RSA, ECC, Ed25519 |

### The Hybrid Approach (How TLS Works)

```
1. Use asymmetric crypto to exchange a symmetric key
2. Use symmetric crypto for actual data (fast!)

Best of both worlds:
- Asymmetric solves key distribution
- Symmetric provides speed
```

### Common File Extensions

| Extension | Contents |
|-----------|----------|
| `.pem` | Base64-encoded, "-----BEGIN...-----" |
| `.der` | Binary encoded |
| `.crt`, `.cer` | Certificate |
| `.key` | Private key |
| `.p12`, `.pfx` | Bundle (cert + key, password protected) |
| `.csr` | Certificate signing request |

---

Next: [Practical Cryptography](./02_PRACTICAL.md) - Real-world applications for robotics and apps
