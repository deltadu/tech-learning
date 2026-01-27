# Practical Cryptography

Real-world crypto applications for robotics and app development.

---

## Table of Contents

1. [Password Storage](#password-storage)
2. [API Authentication](#api-authentication)
3. [Encrypting Data at Rest](#encrypting-data-at-rest)
4. [Secure Communication](#secure-communication)
5. [Robotics Security](#robotics-security)
6. [Mobile App Security](#mobile-app-security)
7. [Crypto Libraries](#crypto-libraries)
8. [Common Mistakes](#common-mistakes)
9. [Quick Recipes](#quick-recipes)

---

## Password Storage

**Never store passwords in plain text. Never.**

### The Right Way

```
1. User creates password: "hunter2"

2. Generate random salt: "x7Kj9mNp..."

3. Hash with slow algorithm:
   hash = argon2id(password="hunter2", salt="x7Kj9mNp...", 
                   time=3, memory=64MB, parallelism=4)

4. Store: "x7Kj9mNp...$argon2id$v=19$m=65536,t=3,p=4$..."
          ─────────── ──────────────────────────────────
             salt              hash output
```

### Password Hashing Algorithms

| Algorithm | Use? | Notes |
|-----------|------|-------|
| **Argon2id** | ✓ Best | Winner of Password Hashing Competition (2015) |
| **bcrypt** | ✓ Good | Proven, widely available |
| **scrypt** | ✓ Good | Memory-hard |
| PBKDF2 | ⚠️ | Only if others unavailable |
| SHA-256 | ❌ | Too fast! Attackers can try billions/sec |
| MD5 | ❌❌ | Never |

### Why Slow is Good

```
Fast hash (SHA-256):
  Attacker can try: 10 billion passwords/second
  8-char password cracked in: minutes

Slow hash (bcrypt, cost=12):
  Attacker can try: ~100 passwords/second
  8-char password cracked in: centuries
```

### Password Verification

```python
# Python with argon2-cffi
from argon2 import PasswordHasher

ph = PasswordHasher()

# Registration
hash = ph.hash("user_password")
# Store 'hash' in database

# Login
try:
    ph.verify(stored_hash, "user_input")
    # Password correct
except argon2.exceptions.VerifyMismatchError:
    # Password wrong
```

---

## API Authentication

### API Keys

Simple but limited:

```
Header: X-API-Key: sk_live_abc123xyz789...

Pros:
✓ Simple to implement
✓ Easy to rotate

Cons:
✗ No expiration (unless you build it)
✗ Can be leaked in logs, URLs
✗ Hard to scope permissions
```

### HMAC Authentication

Prove you have the secret without sending it:

```
Request to sign:
  POST /api/orders
  Content-Type: application/json
  X-Timestamp: 1699123456
  Body: {"item": "widget", "qty": 5}

Create signature string:
  POST\n/api/orders\n1699123456\n{"item":"widget","qty":5}

Sign with HMAC:
  signature = HMAC-SHA256(secret_key, signature_string)
  
Send:
  Authorization: HMAC-SHA256 signature="abc123...", timestamp="1699123456"

Server:
  1. Rebuild signature string from request
  2. Compute HMAC with stored secret
  3. Compare signatures (timing-safe!)
  4. Check timestamp isn't too old (prevent replay)
```

### JWT (JSON Web Tokens)

Self-contained tokens for stateless authentication:

```
eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxMjM0LCJleHAiOjE2OTkxMjM0NTZ9.signature
──────────────────────┬─────────────────────────────────────────────┬──────────
       Header                         Payload                       Signature
  {"alg":"HS256"}           {"user_id":1234,"exp":1699123456}

Header + Payload are just Base64 (NOT encrypted!)
Signature proves token wasn't tampered with
```

**JWT Best Practices:**

| Do | Don't |
|----|-------|
| ✓ Short expiration (15min - 1hr) | ✗ Put secrets in payload |
| ✓ Use RS256 for distributed systems | ✗ Use "none" algorithm |
| ✓ Validate all claims (exp, iss, aud) | ✗ Store sensitive data |
| ✓ Use refresh tokens for long sessions | ✗ Make tokens too long-lived |

**HS256 vs RS256:**

```
HS256 (Symmetric):
  - Same secret signs and verifies
  - Simpler, faster
  - Good for: Single service

RS256 (Asymmetric):  
  - Private key signs, public key verifies
  - Anyone can verify without secret
  - Good for: Microservices, third-party verification
```

### OAuth 2.0 / OpenID Connect

For delegated authorization (e.g., "Login with Google"):

```
┌──────────────────────────────────────────────────────────────┐
│ OAuth 2.0 Authorization Code Flow                            │
│                                                              │
│  User          Your App         Auth Server      Resource    │
│   │               │                  │              │        │
│   │──Login────────▶                  │              │        │
│   │               │                  │              │        │
│   │◀─────────Redirect to auth────────│              │        │
│   │                                  │              │        │
│   │──────────Login at auth server────▶              │        │
│   │                                  │              │        │
│   │◀────Redirect with auth code──────│              │        │
│   │                                  │              │        │
│   │               │──Exchange code───▶              │        │
│   │               │  for tokens      │              │        │
│   │               │◀─Access token────│              │        │
│   │               │                  │              │        │
│   │               │──────────────────Access API─────▶        │
│   │               │◀─────────────────Data───────────│        │
└──────────────────────────────────────────────────────────────┘
```

---

## Encrypting Data at Rest

### File Encryption

**Symmetric (for your own files):**

```bash
# Using age (modern, simple)
age -p -o secret.txt.age secret.txt

# Using GPG
gpg -c secret.txt  # symmetric
gpg -e -r recipient@email.com secret.txt  # asymmetric
```

**For programmatic encryption:**

```python
# Python with cryptography library
from cryptography.fernet import Fernet

# Generate key (store securely!)
key = Fernet.generate_key()
f = Fernet(key)

# Encrypt
encrypted = f.encrypt(b"secret data")

# Decrypt
decrypted = f.decrypt(encrypted)
```

### Database Field Encryption

```
┌─────────────────────────────────────────────────────────────┐
│ Application-Level Encryption                                │
│                                                             │
│   App encrypts sensitive fields before storing:             │
│                                                             │
│   users table:                                              │
│   ┌────────────┬───────────────────────────────────────┐   │
│   │ id         │ 12345                                 │   │
│   │ email      │ user@example.com (plain)              │   │
│   │ ssn        │ AES-GCM(iv + ciphertext + tag)       │   │
│   │ credit_card│ AES-GCM(iv + ciphertext + tag)       │   │
│   └────────────┴───────────────────────────────────────┘   │
│                                                             │
│   Benefits:                                                 │
│   ✓ Data protected even if database stolen                  │
│   ✓ Fine-grained control over what's encrypted              │
│                                                             │
│   Challenges:                                               │
│   ✗ Can't query encrypted fields efficiently                │
│   ✗ Key management complexity                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Management

**The hardest problem in crypto: where do you store the key?**

| Approach | Security | Complexity |
|----------|----------|------------|
| Environment variable | Low | Low |
| Config file (encrypted) | Medium | Medium |
| Cloud KMS (AWS/GCP/Azure) | High | Medium |
| Hardware Security Module (HSM) | Very High | High |

**Key hierarchy pattern:**

```
Master Key (in HSM/KMS, never leaves)
     │
     └─▶ Data Encryption Key (DEK)
            │
            └─▶ Encrypted DEK stored with data
                (re-encrypt DEK when rotating master)
```

---

## Secure Communication

### When TLS Isn't Enough

TLS protects data in transit, but consider:

| Threat | TLS Alone | Additional Measure |
|--------|-----------|-------------------|
| MITM on network | ✓ Protected | - |
| Compromised CA | ✗ Vulnerable | Certificate pinning |
| Server compromise | ✗ Vulnerable | End-to-end encryption |
| Insider at recipient | ✗ Vulnerable | Client-side encryption |

### Certificate Pinning

Don't trust any CA - only trust YOUR certificate:

```swift
// iOS example
let pinnedCertificates: [SecCertificate] = // load your cert

func urlSession(_ session: URLSession, 
                didReceive challenge: URLAuthenticationChallenge,
                completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
    
    guard let serverCert = challenge.protectionSpace.serverTrust.flatMap({ 
        SecTrustGetCertificateAtIndex($0, 0) 
    }) else {
        completionHandler(.cancelAuthenticationChallenge, nil)
        return
    }
    
    // Compare server cert to pinned cert
    if pinnedCertificates.contains(serverCert) {
        completionHandler(.useCredential, URLCredential(trust: challenge.protectionSpace.serverTrust!))
    } else {
        completionHandler(.cancelAuthenticationChallenge, nil)
    }
}
```

**Pinning strategies:**
- **Pin certificate**: Most secure, but must update app when cert rotates
- **Pin public key**: Survives cert renewal if key stays same
- **Pin CA**: Less secure but more flexible

### End-to-End Encryption

Data encrypted on sender, decrypted only by recipient:

```
┌─────────────────────────────────────────────────────────────┐
│ E2E Encryption (Signal Protocol style)                      │
│                                                             │
│  Alice                 Server               Bob              │
│    │                     │                   │              │
│    │                     │   ◀──Bob's public │              │
│    │  Get Bob's key────▶ │      key          │              │
│    │  ◀──Bob's public────│                   │              │
│    │                     │                   │              │
│    │  Encrypt message    │                   │              │
│    │  with Bob's key     │                   │              │
│    │                     │                   │              │
│    │  ──Encrypted msg───▶│───Encrypted msg──▶│              │
│    │                     │                   │              │
│    │                     │      Decrypt with │              │
│    │                     │      Bob's private│              │
│    │                     │                   │              │
│                                                             │
│  Server can store/forward but CANNOT read message           │
└─────────────────────────────────────────────────────────────┘
```

---

## Robotics Security

### Challenges in Robotics

| Challenge | Consideration |
|-----------|---------------|
| Real-time constraints | Crypto can't add too much latency |
| Limited compute | Embedded systems may lack hardware crypto |
| Physical access | Attacker may have device in hand |
| Long deployment | Keys may need to last years |
| OTA updates | Need secure update mechanism |

### Securing Robot Communication

**ROS 2 SROS2 (Secure ROS 2):**

```
┌─────────────────────────────────────────────────────────────┐
│ SROS2 Security                                              │
│                                                             │
│   ┌─────────┐      DDS Security       ┌─────────┐          │
│   │ Node A  │◀═══════════════════════▶│ Node B  │          │
│   └─────────┘   (TLS-like in DDS)     └─────────┘          │
│                                                             │
│   Each node has:                                            │
│   • Certificate (identity)                                  │
│   • Private key                                             │
│   • Permissions file (access control)                       │
│                                                             │
│   Provides:                                                 │
│   ✓ Authentication (nodes prove identity)                   │
│   ✓ Encryption (topic data encrypted)                       │
│   ✓ Access control (who can pub/sub to what)                │
└─────────────────────────────────────────────────────────────┘
```

**Setting up SROS2:**

```bash
# Generate keystore
ros2 security create_keystore ~/sros2_keystore

# Create keys for a node
ros2 security create_key ~/sros2_keystore /my_robot/camera_node

# Run with security
export ROS_SECURITY_KEYSTORE=~/sros2_keystore
export ROS_SECURITY_ENABLE=true
ros2 run my_package my_node
```

### Lightweight Crypto for Embedded

When AES is too heavy:

| Algorithm | Use Case |
|-----------|----------|
| ChaCha20-Poly1305 | Software-only (no AES hardware) |
| PRESENT | Ultra-lightweight block cipher |
| ASCON | Lightweight AEAD (NIST selection) |

### Secure Boot

Ensure only authorized firmware runs:

```
┌─────────────────────────────────────────────────────────────┐
│ Secure Boot Chain                                           │
│                                                             │
│   ROM Bootloader (immutable, in silicon)                    │
│        │                                                    │
│        │ verifies signature                                 │
│        ▼                                                    │
│   First Stage Bootloader                                    │
│        │                                                    │
│        │ verifies signature                                 │
│        ▼                                                    │
│   Second Stage / OS Kernel                                  │
│        │                                                    │
│        │ verifies signature                                 │
│        ▼                                                    │
│   Application                                               │
│                                                             │
│   Each stage verifies the next before executing             │
│   Root of trust: public key burned into hardware            │
└─────────────────────────────────────────────────────────────┘
```

### Firmware Updates

Secure OTA (Over-The-Air) updates:

```
1. Sign firmware image with private key (kept offline!)

2. Device downloads update:
   ┌─────────────────────────────────────────────┐
   │ Firmware Image                              │
   │ ┌─────────────────────────────────────────┐ │
   │ │ Header: version, size, hash             │ │
   │ ├─────────────────────────────────────────┤ │
   │ │ Firmware binary                         │ │
   │ ├─────────────────────────────────────────┤ │
   │ │ Signature (of header + binary)          │ │
   │ └─────────────────────────────────────────┘ │
   └─────────────────────────────────────────────┘

3. Device verifies:
   - Signature valid? (using embedded public key)
   - Version > current? (prevent downgrade)
   - Hash matches?

4. Install only if all checks pass
```

---

## Mobile App Security

### Keychain / Keystore

**Never store secrets in code or UserDefaults/SharedPreferences!**

```swift
// iOS Keychain
import Security

func saveToKeychain(key: String, data: Data) throws {
    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrAccount as String: key,
        kSecValueData as String: data,
        kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    ]
    
    let status = SecItemAdd(query as CFDictionary, nil)
    guard status == errSecSuccess else { throw KeychainError.saveFailed }
}
```

```kotlin
// Android Keystore
val keyStore = KeyStore.getInstance("AndroidKeyStore")
keyStore.load(null)

val keyGenerator = KeyGenerator.getInstance(
    KeyProperties.KEY_ALGORITHM_AES,
    "AndroidKeyStore"
)

keyGenerator.init(
    KeyGenParameterSpec.Builder("my_key",
        KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
        .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
        .setUserAuthenticationRequired(true)  // Require biometric/PIN
        .build()
)
```

### Secure Storage Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ What to store where                                         │
│                                                             │
│   Most Secure                                               │
│   ────────────                                              │
│   │ Hardware-backed Keystore                                │
│   │   - Cryptographic keys                                  │
│   │   - Never leave secure enclave                          │
│   │                                                         │
│   │ Keychain/Keystore (software)                            │
│   │   - Auth tokens                                         │
│   │   - API keys                                            │
│   │   - User credentials                                    │
│   │                                                         │
│   │ Encrypted local storage                                 │
│   │   - Sensitive user data                                 │
│   │   - Cached private content                              │
│   │                                                         │
│   │ Regular storage (UserDefaults, SharedPrefs)             │
│   │   - Non-sensitive preferences                           │
│   │   - UI state                                            │
│   ▼                                                         │
│   Least Secure                                              │
└─────────────────────────────────────────────────────────────┘
```

### App Attestation

Prove your app is genuine (not tampered/emulated):

```
iOS: DeviceCheck / App Attest
Android: SafetyNet / Play Integrity API

Server can verify:
- Request comes from genuine app
- App wasn't modified
- Device isn't rooted/jailbroken (optional)
```

### Obfuscation (Defense in Depth)

Not crypto, but slows down reverse engineering:

```
Code obfuscation:
- Rename functions: authenticate() → a1b2c3()
- Control flow flattening
- String encryption

Runtime protection:
- Jailbreak/root detection
- Debugger detection
- Tampering detection

Remember: Obfuscation is NOT security. 
          Determined attacker will eventually succeed.
          It just raises the bar.
```

---

## Crypto Libraries

### Choose Wisely

| Language | Recommended | Avoid |
|----------|-------------|-------|
| Python | `cryptography`, `PyNaCl` | `pycrypto` (abandoned) |
| JavaScript | `libsodium.js`, Web Crypto API | `crypto-js` |
| Go | `crypto/*` (stdlib), `x/crypto` | - |
| Rust | `ring`, `rustcrypto/*` | - |
| C/C++ | libsodium, OpenSSL | Rolling your own |
| iOS | CryptoKit, Security.framework | CommonCrypto (low-level) |
| Android | javax.crypto + Keystore | Spongy Castle (outdated) |

### libsodium

Easy-to-use, hard to misuse:

```python
# pip install pynacl
from nacl.secret import SecretBox
from nacl.utils import random

# Symmetric encryption
key = random(SecretBox.KEY_SIZE)  # 32 bytes
box = SecretBox(key)

encrypted = box.encrypt(b"secret message")
decrypted = box.decrypt(encrypted)
```

### Python `cryptography` Library

More control, more rope to hang yourself:

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

# AES-GCM
key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)

nonce = os.urandom(12)  # MUST be unique per encryption
ciphertext = aesgcm.encrypt(nonce, b"secret", b"associated data")
plaintext = aesgcm.decrypt(nonce, ciphertext, b"associated data")
```

---

## Common Mistakes

### 1. Using Encryption Without Authentication

```
❌ AES-CBC without HMAC
   Attacker can flip bits in ciphertext
   
✓ AES-GCM (authenticated encryption)
   Detects any tampering
```

### 2. Reusing IV/Nonce

```
❌ Always using IV = 0
   OR using same IV twice with same key
   
   With CTR/GCM mode: XOR of plaintexts leaked!
   
✓ Random IV for each encryption
   OR counter-based IV (carefully managed)
```

### 3. Timing Attacks

```python
# ❌ Vulnerable comparison
if user_signature == computed_signature:
    # Attacker can time how many bytes matched

# ✓ Constant-time comparison
import hmac
if hmac.compare_digest(user_signature, computed_signature):
    # Same time regardless of how many bytes match
```

### 4. Weak Key Derivation

```python
# ❌ Direct use of password as key
key = password.encode()[:32]  # Weak!

# ✓ Proper key derivation
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
kdf = PBKDF2HMAC(algorithm=SHA256(), length=32, salt=salt, iterations=600000)
key = kdf.derive(password.encode())
```

### 5. Rolling Your Own Crypto

```
❌ "I'll just XOR with a key"
❌ "I invented a new algorithm"
❌ "I'll make it more secure by encrypting twice with different algorithms"

✓ Use well-vetted libraries
✓ Use standard algorithms
✓ Follow established patterns
```

### 6. Hardcoded Secrets

```kotlin
// ❌ In source code
val API_KEY = "sk_live_abc123..."

// ❌ In config file committed to git
config.json: {"api_key": "sk_live_abc123..."}

// ✓ Environment variable (not in code)
// ✓ Secret manager (AWS Secrets Manager, Vault)
// ✓ Keychain/Keystore (mobile)
```

---

## Quick Recipes

### Generate a Random Key

```python
import os
key = os.urandom(32)  # 256-bit key
```

### Hash a Password (for storage)

```python
from argon2 import PasswordHasher
ph = PasswordHasher()
hash = ph.hash("password")
```

### Create an HMAC

```python
import hmac
import hashlib

signature = hmac.new(
    key=b"secret",
    msg=b"message",
    digestmod=hashlib.sha256
).hexdigest()
```

### Encrypt Data (symmetric)

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()  # Save this!
f = Fernet(key)
encrypted = f.encrypt(b"secret")
decrypted = f.decrypt(encrypted)
```

### Sign Data (asymmetric)

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

signature = private_key.sign(b"message")
public_key.verify(signature, b"message")  # Raises if invalid
```

### Verify a JWT

```python
import jwt

try:
    payload = jwt.decode(token, public_key, algorithms=["RS256"])
    # Valid token
except jwt.ExpiredSignatureError:
    # Token expired
except jwt.InvalidTokenError:
    # Invalid token
```

---

## Key Takeaways

1. **Don't roll your own crypto** - use established libraries
2. **Use authenticated encryption** - AES-GCM or ChaCha20-Poly1305
3. **Never reuse nonces** - random or counter, but unique per key
4. **Hash passwords properly** - Argon2id or bcrypt, not SHA
5. **Protect your keys** - HSM > KMS > encrypted files > env vars
6. **Defense in depth** - TLS + app-level encryption + signing
7. **Plan for key rotation** - it will happen eventually
8. **Constant-time comparisons** - for any secret comparison
