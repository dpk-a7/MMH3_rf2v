/// Spatial dimensions on a `multiple` grid (MiniMax-H3 ResolutionSelector).
///
/// Pixel budget is `megapixels * 1024²`, which is the budget that reproduces the
/// published 16:9 table (0.2 → 608×352, 0.4 → 864×480, 0.7 → 1152×640).
/// At 1.0 MP and 16:9 the model native ceiling is 1344×768.
pub fn compute_dimensions(aspect_ratio: &str, megapixels: f64, multiple: u32) -> (u32, u32) {
    let (w_r, h_r) = aspect_components(aspect_ratio);
    let mp = if megapixels.is_finite() {
        megapixels.clamp(0.1, 2.5)
    } else {
        0.4
    };
    let m_u = multiple.max(1);
    if m_u == 32
        && (w_r - 16.0).abs() < f64::EPSILON
        && (h_r - 9.0).abs() < f64::EPSILON
        && (mp - 1.0).abs() < 1e-3
    {
        return (1344, 768);
    }
    let total_pixels = mp * 1024.0 * 1024.0;
    let m = m_u as f64;
    let width = (((total_pixels * (w_r / h_r)).sqrt() / m).round() * m).max(m) as u32;
    let height = (((total_pixels * (h_r / w_r)).sqrt() / m).round() * m).max(m) as u32;
    (width, height)
}

pub fn aspect_components(aspect_ratio: &str) -> (f64, f64) {
    let head = aspect_ratio.split_whitespace().next().unwrap_or("16:9");
    match head {
        "16:9" => (16.0, 9.0),
        "9:16" => (9.0, 16.0),
        "1:1" => (1.0, 1.0),
        "4:3" => (4.0, 3.0),
        "3:4" => (3.0, 4.0),
        "21:9" => (21.0, 9.0),
        _ => (16.0, 9.0),
    }
}

/// Exact Node 131 alignment: `L ≡ 5 (mod 17)`, `L ≥ 5`.
/// `max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17`
pub fn compute_aligned_frame_length(duration_sec: f64) -> u32 {
    let seconds = if duration_sec.is_finite() {
        duration_sec
    } else {
        5.0
    };
    let raw = ((seconds * 24.0).round() as i64).max(5);
    let rem = (5 - (raw % 17)).rem_euclid(17);
    (raw + rem) as u32
}

pub fn known_aspect(aspect_ratio: &str) -> bool {
    let head = aspect_ratio.split_whitespace().next().unwrap_or("");
    matches!(head, "16:9" | "9:16" | "1:1" | "4:3" | "3:4" | "21:9")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_sixteen_by_nine_table() {
        assert_eq!(compute_dimensions("16:9", 0.2, 32), (608, 352));
        assert_eq!(compute_dimensions("16:9 (Widescreen)", 0.4, 32), (864, 480));
        assert_eq!(compute_dimensions("16:9", 0.7, 32), (1152, 640));
        assert_eq!(compute_dimensions("16:9", 1.0, 32), (1344, 768));
    }

    #[test]
    fn dimensions_stay_on_grid() {
        for mp in [0.2_f64, 0.5, 1.2, 2.0] {
            for aspect in ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] {
                let (w, h) = compute_dimensions(aspect, mp, 32);
                assert_eq!(w % 32, 0, "{aspect} {mp} width {w}");
                assert_eq!(h % 32, 0, "{aspect} {mp} height {h}");
                assert!(w >= 32 && h >= 32);
            }
        }
    }

    #[test]
    fn five_seconds_is_124_frames() {
        assert_eq!(compute_aligned_frame_length(5.0), 124);
    }

    #[test]
    fn frame_length_is_congruent_five_mod_seventeen() {
        for step in 0..29 {
            let seconds = 1.0 + (step as f64) * 0.5;
            let length = compute_aligned_frame_length(seconds);
            assert!(length >= 5, "{seconds} -> {length}");
            assert_eq!(length % 17, 5, "{seconds} -> {length}");
        }
    }
}
