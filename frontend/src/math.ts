export function aspectParts(aspect: string): [number, number] {
  if (aspect.startsWith("16:9")) return [16, 9];
  if (aspect.startsWith("9:16")) return [9, 16];
  if (aspect.startsWith("1:1")) return [1, 1];
  if (aspect.startsWith("4:3")) return [4, 3];
  if (aspect.startsWith("3:4")) return [3, 4];
  if (aspect.startsWith("21:9")) return [21, 9];
  return [16, 9];
}

export function computeSize(aspect: string, megapixels: number): [number, number] {
  const [wR, hR] = aspectParts(aspect);
  if (wR === 16 && hR === 9 && Math.abs(megapixels - 1) < 1e-3) {
    return [1344, 768];
  }
  const pixels = megapixels * 1024 * 1024;
  const width = Math.max(32, Math.round(Math.sqrt(pixels * (wR / hR)) / 32) * 32);
  const height = Math.max(32, Math.round(Math.sqrt(pixels * (hR / wR)) / 32) * 32);
  return [width, height];
}

export function computeDimensions(aspect: string, megapixels: number): string {
  const [width, height] = computeSize(aspect, megapixels);
  return `${width} × ${height}`;
}

export function computeAlignedFrames(seconds: number): number {
  const raw = Math.max(5, Math.round(seconds * 24));
  const rem = ((5 - (raw % 17)) % 17 + 17) % 17;
  return raw + rem;
}
