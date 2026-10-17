import { decode, encode } from "@msgpack/msgpack";

const PAD_BLOCK = 65536;
const HKDF_SALT = new TextEncoder().encode("mmh3-ref2va-salt-v1");
const HKDF_INFO = new TextEncoder().encode("mmh3-ref2va-session-v1");

export type EnvelopeFile = {
  name: string;
  mime: string;
  kind: string;
  data: Uint8Array;
};

export type EnvelopeResult = {
  ok: boolean;
  action: string;
  metadata: Record<string, unknown>;
  files: EnvelopeFile[];
  error?: string;
};

export type GpuInfo = {
  index: number;
  name: string;
  memory_total_mb?: number;
  free_vram_mb?: number;
  total_vram_mb?: number;
  allocated_weight_pct?: number;
  role?: "active_full" | "active_tight_partial" | "bypassed_busy" | string;
};

export type ProgressEvent = {
  job_id: string;
  stage: string;
  progress: number;
  detail: string;
  done: boolean;
  ok: boolean;
  gpus?: GpuInfo[];
};

type HandshakeResponse = {
  session_id: string;
  server_public_key: string;
  alg: string;
};

export class SecureClient {
  private key: CryptoKey | null = null;
  private sessionId: string | null = null;
  private socket: WebSocket | null = null;

  get ready(): boolean {
    return this.key !== null && this.sessionId !== null;
  }

  reset(): void {
    this.key = null;
    this.sessionId = null;
    if (this.socket) {
      this.socket.onmessage = null;
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
  }

  async handshake(): Promise<void> {
    const pair = (await crypto.subtle.generateKey(
      { name: "X25519" } as Algorithm,
      true,
      ["deriveBits"],
    )) as CryptoKeyPair;
    const rawPublic = new Uint8Array(await crypto.subtle.exportKey("raw", pair.publicKey));
    const response = await fetch("/api/v1/crypto/handshake", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_public_key: bytesToB64(rawPublic) }),
    });
    if (!response.ok) {
      throw new Error("handshake_failed");
    }
    const body = (await response.json()) as HandshakeResponse;
    const serverKey = await crypto.subtle.importKey(
      "raw",
      b64ToBytes(body.server_public_key),
      { name: "X25519" } as Algorithm,
      false,
      [],
    );
    const shared = await crypto.subtle.deriveBits(
      { name: "X25519", public: serverKey } as Algorithm,
      pair.privateKey,
      256,
    );
    const hkdfKey = await crypto.subtle.importKey("raw", shared, "HKDF", false, ["deriveKey"]);
    this.key = await crypto.subtle.deriveKey(
      { name: "HKDF", hash: "SHA-256", salt: HKDF_SALT, info: HKDF_INFO },
      hkdfKey,
      { name: "AES-GCM", length: 256 },
      false,
      ["encrypt", "decrypt"],
    );
    this.sessionId = body.session_id;
  }

  async call(
    action: string,
    metadata: Record<string, unknown>,
    files: EnvelopeFile[] = [],
  ): Promise<EnvelopeResult> {
    if (!this.key || !this.sessionId) {
      throw new Error("session_expired");
    }
    const packed = encode({
      action,
      headers: {},
      metadata,
      files,
    });
    const padded = padBlock(toBytes(packed));
    const sealed = await seal(this.key, padded);
    const response = await fetch("/api/v1/envelope", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Session-Id": this.sessionId,
      },
      body: sealed,
    });
    if (!response.ok) {
      throw new Error(response.status === 401 ? "session_expired" : "request_failed");
    }
    const opened = await open(this.key, new Uint8Array(await response.arrayBuffer()));
    const decoded = decode(unpadBlock(opened)) as EnvelopeResult;
    decoded.files = (decoded.files ?? []).map((file) => ({
      ...file,
      data: toBytes(file.data),
    }));
    return decoded;
  }

  connect(onEvent: (event: ProgressEvent) => void): void {
    if (!this.sessionId || !this.key) {
      return;
    }
    if (this.socket) {
      this.socket.onmessage = null;
      this.socket.close();
    }
    const key = this.key;
    const socket = new WebSocket(`wss://${location.host}/api/v1/ws?sid=${this.sessionId}`);
    socket.binaryType = "arraybuffer";
    socket.onmessage = (message) => {
      const data = message.data;
      if (!(data instanceof ArrayBuffer)) {
        return;
      }
      void open(key, new Uint8Array(data))
        .then((plain) => {
          onEvent(decode(plain) as ProgressEvent);
        })
        .catch(() => undefined);
    };
    this.socket = socket;
  }
}

function padBlock(data: Uint8Array): Uint8Array<ArrayBuffer> {
  const raw = 4 + data.length;
  const rem = raw % PAD_BLOCK;
  const size = rem === 0 ? raw : raw + (PAD_BLOCK - rem);
  const buffer = new ArrayBuffer(size);
  const out = new Uint8Array(buffer);
  new DataView(buffer).setUint32(0, data.length, true);
  out.set(data, 4);
  return out;
}

function unpadBlock(data: Uint8Array): Uint8Array {
  if (data.length < 4 || data.length % PAD_BLOCK !== 0) {
    throw new Error("bad_envelope");
  }
  const len = new DataView(data.buffer, data.byteOffset, 4).getUint32(0, true);
  if (len > data.length - 4) {
    throw new Error("bad_envelope");
  }
  return data.slice(4, 4 + len);
}

async function seal(key: CryptoKey, plain: Uint8Array): Promise<Uint8Array<ArrayBuffer>> {
  const nonce = crypto.getRandomValues(new Uint8Array(new ArrayBuffer(12)));
  const cipher = new Uint8Array(
    await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce }, key, owned(plain)),
  );
  const out = new Uint8Array(new ArrayBuffer(nonce.length + cipher.length));
  out.set(nonce, 0);
  out.set(cipher, nonce.length);
  return out;
}

async function open(key: CryptoKey, packed: Uint8Array): Promise<Uint8Array> {
  if (packed.length < 28) {
    throw new Error("bad_envelope");
  }
  const plain = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: packed.slice(0, 12) },
    key,
    packed.slice(12),
  );
  return new Uint8Array(plain);
}

function owned(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const copy = new Uint8Array(new ArrayBuffer(bytes.byteLength));
  copy.set(bytes);
  return copy;
}

function toBytes(value: unknown): Uint8Array<ArrayBuffer> {
  if (value instanceof Uint8Array) {
    return owned(value);
  }
  if (value instanceof ArrayBuffer) {
    return owned(new Uint8Array(value));
  }
  if (ArrayBuffer.isView(value)) {
    return owned(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
  }
  return new Uint8Array(new ArrayBuffer(0));
}

function bytesToB64(bytes: Uint8Array): string {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
}

function b64ToBytes(value: string): Uint8Array<ArrayBuffer> {
  const binary = atob(value);
  const out = new Uint8Array(new ArrayBuffer(binary.length));
  for (let index = 0; index < binary.length; index += 1) {
    out[index] = binary.charCodeAt(index);
  }
  return out;
}
