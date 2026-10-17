use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use hkdf::Hkdf;
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::Sha256;
use x25519_dalek::{EphemeralSecret, PublicKey};
use zeroize::{Zeroize, ZeroizeOnDrop};

pub const PAD_BLOCK: usize = 65_536;
pub const HKDF_SALT: &[u8] = b"mmh3-ref2va-salt-v1";
pub const HKDF_INFO: &[u8] = b"mmh3-ref2va-session-v1";
pub const SESSION_ALG: &str = "X25519-HKDF-SHA256-AES-256-GCM";

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SessionKey {
    bytes: [u8; 32],
}

impl SessionKey {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

#[derive(Debug)]
pub enum CryptoError {
    Key,
    Seal,
    Open,
    Pad,
    Handshake,
}

pub fn decode_public_key(b64: &str) -> Result<[u8; 32], CryptoError> {
    let raw = STANDARD
        .decode(b64.trim().as_bytes())
        .map_err(|_| CryptoError::Handshake)?;
    if raw.len() != 32 {
        return Err(CryptoError::Handshake);
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&raw);
    Ok(key)
}

pub fn encode_public_key(key: &[u8; 32]) -> String {
    STANDARD.encode(key)
}

pub fn handshake(client_public: &[u8; 32]) -> Result<([u8; 32], SessionKey), CryptoError> {
    let secret = EphemeralSecret::random_from_rng(OsRng);
    let server_public = PublicKey::from(&secret);
    let shared = secret.diffie_hellman(&PublicKey::from(*client_public));
    let session = derive_session_key(shared.as_bytes())?;
    Ok((*server_public.as_bytes(), SessionKey::new(session)))
}

pub fn derive_session_key(shared: &[u8; 32]) -> Result<[u8; 32], CryptoError> {
    let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT), shared);
    let mut okm = [0u8; 32];
    hk.expand(HKDF_INFO, &mut okm)
        .map_err(|_| CryptoError::Key)?;
    Ok(okm)
}

pub fn pad_block(data: &[u8]) -> Result<Vec<u8>, CryptoError> {
    let len = u32::try_from(data.len()).map_err(|_| CryptoError::Pad)?;
    let mut out = Vec::with_capacity(PAD_BLOCK);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(data);
    let rem = out.len() % PAD_BLOCK;
    if rem != 0 {
        out.resize(out.len() + (PAD_BLOCK - rem), 0);
    }
    Ok(out)
}

pub fn unpad_block(data: &[u8]) -> Result<Vec<u8>, CryptoError> {
    if data.len() < 4 || data.len() % PAD_BLOCK != 0 {
        return Err(CryptoError::Pad);
    }
    let len = u32::from_le_bytes(data[0..4].try_into().map_err(|_| CryptoError::Pad)?) as usize;
    if len > data.len() - 4 {
        return Err(CryptoError::Pad);
    }
    Ok(data[4..4 + len].to_vec())
}

/// Wire format: 12-byte nonce || AES-256-GCM ciphertext (tag included).
pub fn seal(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError::Key)?;
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ct = cipher
        .encrypt(nonce, plaintext)
        .map_err(|_| CryptoError::Seal)?;
    let mut out = Vec::with_capacity(12 + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    nonce_bytes.zeroize();
    Ok(out)
}

pub fn open(key: &[u8; 32], packed: &[u8]) -> Result<Vec<u8>, CryptoError> {
    if packed.len() < 12 + 16 {
        return Err(CryptoError::Open);
    }
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError::Key)?;
    let nonce = Nonce::from_slice(&packed[..12]);
    cipher
        .decrypt(nonce, &packed[12..])
        .map_err(|_| CryptoError::Open)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_roundtrip_and_block_alignment() {
        let payload = b"ref2va-envelope";
        let padded = pad_block(payload).unwrap();
        assert_eq!(padded.len() % PAD_BLOCK, 0);
        assert_eq!(unpad_block(&padded).unwrap(), payload);
    }

    #[test]
    fn both_sides_derive_the_same_session_key() {
        let client_secret = EphemeralSecret::random_from_rng(OsRng);
        let client_public = PublicKey::from(&client_secret);
        let (server_public, server_key) = handshake(client_public.as_bytes()).unwrap();
        let shared = client_secret.diffie_hellman(&PublicKey::from(server_public));
        let client_key = derive_session_key(shared.as_bytes()).unwrap();
        let sealed = seal(&client_key, b"picture-bytes").unwrap();
        let opened = open(server_key.as_bytes(), &sealed).unwrap();
        assert_eq!(opened, b"picture-bytes");
    }
}
