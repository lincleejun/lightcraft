//! TIFF decode via the `tiff` crate: 1/2/4/8/16/32/64-bit integer, 16/32/64-bit float, gray/RGB/CMYK,
//! alpha (associated or not), ICC (34675), XMP (700), orientation (274). First image only. Palette
//! (indexed-colour) images are refused by the `tiff` crate itself, on every read path.

use crate::convert::{Buf, Meta, Model, Raw, check_size, finish};
use crate::{DecodeOptions, Decoded, Error, Format, Result};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

const F: Format = Format::Tiff;

fn err(e: impl std::fmt::Display) -> Error {
    Error::Malformed(F, e.to_string())
}

pub(crate) fn decode(bytes: &[u8], opts: &DecodeOptions) -> Result<Decoded> {
    let mut d = Decoder::new(std::io::Cursor::new(bytes)).map_err(err)?;
    let compression = d.get_tag_u32(Tag::Compression).ok();
    let mut d = d.with_limits(limits(opts, bytes.len(), compression));
    let (w, h) = d.dimensions().map_err(err)?;
    check_size(F, w as u64, h as u64, opts)?;
    let ct = d.colortype().map_err(err)?;

    let icc = tag_bytes(&mut d, Tag::IccProfile);
    let xmp = tag_bytes(&mut d, Tag::from_u16_exhaustive(700)).map(|v| String::from_utf8_lossy(&v).trim_end_matches('\0').to_string());
    let orientation = d.get_tag_u32(Tag::Orientation).ok().map(|v| v as u16);
    let planar = d.get_tag_u32(Tag::PlanarConfiguration).ok() == Some(2);
    let extra = d.get_tag_u16_vec(Tag::ExtraSamples).ok().unwrap_or_default();
    let colormap = d.get_tag_u16_vec(Tag::ColorMap).ok();

    let (model, n_color, n_samples, bits) = layout(ct, compression, planar)?;
    let has_alpha = n_samples > n_color && !matches!(ct, ColorType::Palette(_));
    let premultiplied = has_alpha && extra.first() == Some(&1);

    let res = d.read_image().map_err(err)?;
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let mut buf = match res {
        DecodingResult::U8(v) if bits < 8 => Buf::U8(unpack(&v, wu, hu, n_samples, bits)?),
        DecodingResult::U8(v) => Buf::U8(v),
        DecodingResult::U16(v) => Buf::U16(v),
        // Rare integer widths: keep 16 significant bits (plenty for display-referred data).
        DecodingResult::U32(v) => Buf::U16(v.into_iter().map(|x| (x >> 16) as u16).collect()),
        DecodingResult::U64(v) => Buf::U16(v.into_iter().map(|x| (x >> 48) as u16).collect()),
        DecodingResult::I8(v) => Buf::U8(v.into_iter().map(|x| (x.max(0) as u8) << 1).collect()),
        DecodingResult::I16(v) => Buf::U16(v.into_iter().map(|x| (x.max(0) as u16) << 1).collect()),
        DecodingResult::I32(v) => Buf::U16(v.into_iter().map(|x| (x.max(0) >> 15) as u16).collect()),
        DecodingResult::I64(v) => Buf::U16(v.into_iter().map(|x| (x.max(0) >> 47) as u16).collect()),
        DecodingResult::F16(v) => Buf::F32(v.into_iter().map(|x| x.to_f32()).collect()),
        DecodingResult::F32(v) => Buf::F32(v),
        DecodingResult::F64(v) => Buf::F32(v.into_iter().map(|x| x as f32).collect()),
    };
    let bit_depth = bits;

    if planar && n_samples > 1 {
        buf = interleave(buf, n, n_samples);
    }

    // Palette → 16-bit RGB.
    if let ColorType::Palette(b) = ct {
        let map = colormap.ok_or_else(|| err("palette image without ColorMap"))?;
        let entries = 1usize << b.min(16);
        if map.len() < 3 * entries {
            return Err(err("short ColorMap"));
        }
        let idx: Vec<usize> = match &buf {
            Buf::U8(v) => v.iter().map(|&i| i as usize).collect(),
            Buf::U16(v) => v.iter().map(|&i| i as usize).collect(),
            Buf::F32(_) => return Err(err("float palette")),
        };
        let mut out = Vec::with_capacity(n * 3);
        for &i in idx.iter().take(n) {
            let i = i.min(entries - 1);
            out.extend_from_slice(&[map[i], map[entries + i], map[2 * entries + i]]);
        }
        buf = Buf::U16(out);
        let raw = Raw { width: wu, height: hu, model: Model::Rgb, alpha: false, premultiplied: false, buf, bit_depth };
        let meta = Meta { icc, xmp, orientation, ..Default::default() };
        return finish(F, raw, meta, (w, h), opts);
    }

    // WhiteIsZero needs nothing here: the `tiff` crate inverts those samples as it reads them (and
    // refuses WhiteIsZero layouts it can't invert), so inverting again would show a negative.

    // Drop extra samples beyond colour + one alpha.
    let keep = n_color + has_alpha as usize;
    if n_samples != keep {
        buf = select_channels(buf, n, n_samples, keep);
    }
    let raw = Raw { width: wu, height: hu, model, alpha: has_alpha, premultiplied, buf, bit_depth };
    let meta = Meta { icc, xmp, orientation, ..Default::default() };
    finish(F, raw, meta, (w, h), opts)
}

/// JPEG-in-TIFF (compression 7): the `tiff` crate copies each compressed chunk into memory before
/// decoding it, so for it a chunk's declared size is an allocation, not just a read.
const COMPRESSION_JPEG: u32 = 7;

/// Decoder limits for a file of `file_len` bytes stored with `compression`. The image buffer
/// follows `max_pixels` (the dimensions are checked against it before anything is read). One strip
/// or tile may claim as many bytes as the whole file when the method streams (none, LZW, Deflate,
/// PackBits, CCITT): Lightroom and Photoshop write uncompressed exports as a single strip, 132 MiB
/// for a 16-bit 23 MP photo, and the crate's default of 128 MiB refused them before reading a byte
/// ("decoder limits exceeded", #1983). The default stays the floor, so nothing that decoded before
/// is refused now (a small file whose writer overstated a count is tolerated as before), and a
/// count larger than both the floor and the file is still refused. JPEG chunks keep the default:
/// the crate copies them, and a hostile count would copy most of the file again.
fn limits(opts: &DecodeOptions, file_len: usize, compression: Option<u32>) -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = opts.max_pixels.saturating_mul(4 * 8).min(usize::MAX as u64) as usize;
    if compression != Some(COMPRESSION_JPEG) {
        limits.intermediate_buffer_size = limits.intermediate_buffer_size.max(file_len);
    }
    limits.ifd_value_size = 32 << 20;
    limits
}

/// The sample layout of a colour type we can decode: (model, colour samples, samples per pixel,
/// bits per sample).
fn layout(ct: ColorType, compression: Option<u32>, planar: bool) -> Result<(Model, usize, usize, u8)> {
    let l = match ct {
        ColorType::Gray(b) => (Model::Gray, 1, 1, b),
        ColorType::GrayA(b) => (Model::Gray, 1, 2, b),
        ColorType::RGB(b) => (Model::Rgb, 3, 3, b),
        ColorType::RGBA(b) => (Model::Rgb, 3, 4, b),
        ColorType::CMYK(b) => (Model::Cmyk, 4, 4, b),
        ColorType::CMYKA(b) => (Model::Cmyk, 4, 5, b),
        ColorType::Palette(b) => (Model::Rgb, 1, 1, b),
        // JPEG-compressed YCbCr is converted to RGB by the JPEG decoder.
        ColorType::YCbCr(b) if compression == Some(7) => (Model::Rgb, 3, 3, b),
        ColorType::Multiband { bit_depth, num_samples } if num_samples >= 3 => (Model::Rgb, 3, num_samples as usize, bit_depth),
        ColorType::Multiband { bit_depth, num_samples } if num_samples >= 1 => (Model::Gray, 1, num_samples as usize, bit_depth),
        other => return Err(Error::Unsupported(F, unsupported_name(other))),
    };
    if l.3 < 8 && planar && l.2 > 1 {
        return Err(Error::Unsupported(F, "planar TIFF with samples narrower than a byte"));
    }
    Ok(l)
}

/// Stored dimensions and orientation from the first IFD, without reading the image data. Refused,
/// as a decode would refuse it: a layout or compression we can't decode, or strips/tiles that
/// start past the end of the file (truncated). The compressed data itself is not checked.
pub(crate) fn header(bytes: &[u8]) -> Result<(u32, u32, u16)> {
    let mut d = Decoder::new(std::io::Cursor::new(bytes)).map_err(err)?;
    let compression = d.get_tag_u32(Tag::Compression).ok();
    let mut d = d.with_limits(limits(&DecodeOptions::default(), bytes.len(), compression));
    let (w, h) = d.dimensions().map_err(err)?;
    check_size(F, w as u64, h as u64, &DecodeOptions::default())?;
    let ct = d.colortype().map_err(err)?;
    let planar = d.get_tag_u32(Tag::PlanarConfiguration).ok() == Some(2);
    layout(ct, compression, planar)?;
    // what this build of the `tiff` crate decompresses: none, CCITT G4, LZW, JPEG, Deflate, PackBits
    if !matches!(compression.unwrap_or(1), 1 | 4 | 5 | 7 | 8 | 32773 | 32946) {
        return Err(Error::Unsupported(F, "compression method"));
    }
    let chunks = |offsets: Tag, counts: Tag, d: &mut Decoder<std::io::Cursor<&[u8]>>| {
        Some((d.get_tag_u64_vec(offsets).ok()?, d.get_tag_u64_vec(counts).ok()?))
    };
    let (offsets, counts) = chunks(Tag::StripOffsets, Tag::StripByteCounts, &mut d)
        .or_else(|| chunks(Tag::TileOffsets, Tag::TileByteCounts, &mut d))
        .ok_or_else(|| err("no image data"))?;
    // Only where each chunk starts: some writers overstate StripByteCounts, and the `tiff` crate
    // decodes such files (it stops once it has the pixels), so a chunk's end proves nothing.
    let len = bytes.len() as u64;
    if offsets.is_empty() || offsets.iter().zip(&counts).any(|(&o, &c)| c > 0 && o >= len) {
        return Err(err("truncated: image data past the end of the file"));
    }
    let orientation = d.get_tag_u32(Tag::Orientation).ok().map(|v| v as u16).filter(|o| (1..=8).contains(o)).unwrap_or(1);
    Ok((w, h, orientation))
}

/// A BYTE/UNDEFINED tag as bytes (the decoder may surface unknown byte tags as unsigned values).
fn tag_bytes<R: std::io::Read + std::io::Seek>(d: &mut Decoder<R>, tag: Tag) -> Option<Vec<u8>> {
    use tiff::decoder::ifd::Value;
    fn byte(v: &Value) -> Option<u8> {
        match v {
            Value::Byte(b) => Some(*b),
            Value::Unsigned(u) => u8::try_from(*u).ok(),
            Value::Short(u) => u8::try_from(*u).ok(),
            Value::SignedByte(b) => Some(*b as u8),
            _ => None,
        }
    }
    match d.find_tag(tag).ok()?? {
        Value::List(v) => v.iter().map(byte).collect(),
        Value::Ascii(s) => Some(s.into_bytes()),
        v => byte(&v).map(|b| vec![b]),
    }
}

fn unsupported_name(ct: ColorType) -> &'static str {
    match ct {
        ColorType::Lab(_) => "CIE L*a*b* TIFF",
        ColorType::YCbCr(_) => "uncompressed YCbCr TIFF",
        _ => "colour type",
    }
}

/// Samples narrower than a byte (1-bit bilevel and fax scans, 2- and 4-bit gray) arrive packed, most
/// significant bits first, each row padded to a whole byte. Spread them out one per byte, scaled to
/// 0–255 like any 8-bit sample.
fn unpack(packed: &[u8], width: usize, height: usize, samples: usize, bits: u8) -> Result<Vec<u8>> {
    if !matches!(bits, 1 | 2 | 4) {
        return Err(Error::Unsupported(F, "TIFF samples of 3, 5, 6 or 7 bits"));
    }
    let bits = bits as usize;
    let per_row = width.checked_mul(samples).ok_or_else(|| err("row too wide"))?;
    let row_bytes = per_row.checked_mul(bits).ok_or_else(|| err("row too wide"))?.div_ceil(8);
    if row_bytes == 0 || packed.len() / row_bytes < height {
        return Err(err("sample buffer too short"));
    }
    let max = (1u8 << bits) - 1;
    let scale = 255 / max; // 255, 85, 17: exact
    let mut out = Vec::with_capacity(per_row.saturating_mul(height));
    for row in packed.chunks_exact(row_bytes).take(height) {
        let values = row.iter().flat_map(|&b| (1..=8 / bits).map(move |i| (b >> (8 - bits * i)) & max));
        out.extend(values.take(per_row).map(|v| v * scale));
    }
    Ok(out)
}

fn interleave(buf: Buf, n: usize, s: usize) -> Buf {
    fn go<T: Copy + Default>(v: Vec<T>, n: usize, s: usize) -> Vec<T> {
        if v.len() < n * s {
            return v;
        }
        let mut out = vec![T::default(); n * s];
        for c in 0..s {
            for i in 0..n {
                out[i * s + c] = v[c * n + i];
            }
        }
        out
    }
    match buf {
        Buf::U8(v) => Buf::U8(go(v, n, s)),
        Buf::U16(v) => Buf::U16(go(v, n, s)),
        Buf::F32(v) => Buf::F32(go(v, n, s)),
    }
}

fn select_channels(buf: Buf, n: usize, s: usize, keep: usize) -> Buf {
    fn go<T: Copy>(v: Vec<T>, n: usize, s: usize, keep: usize) -> Vec<T> {
        if v.len() < n * s {
            return v;
        }
        let mut out = Vec::with_capacity(n * keep);
        for px in v.chunks_exact(s).take(n) {
            out.extend_from_slice(&px[..keep]);
        }
        out
    }
    match buf {
        Buf::U8(v) => Buf::U8(go(v, n, s, keep)),
        Buf::U16(v) => Buf::U16(go(v, n, s, keep)),
        Buf::F32(v) => Buf::F32(go(v, n, s, keep)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A little-endian classic TIFF, `w` × `h` RGB at `bits` per sample, uncompressed, as one strip of
    /// `declared` bytes (the payload is as long as the image really is, filled with `fill`).
    fn single_strip_rgb(w: u32, h: u32, bits: u16, declared: u32, fill: u8) -> Vec<u8> {
        let entries: [(u16, u16, u32, u32); 11] = [
            (256, 4, 1, w),
            (257, 4, 1, h),
            (258, 3, 3, 0), // BitsPerSample → offset, patched below
            (259, 3, 1, 1), // uncompressed
            (262, 3, 1, 2), // RGB
            (273, 4, 1, 0), // StripOffsets, patched below
            (277, 3, 1, 3),
            (278, 4, 1, h), // one strip
            (279, 4, 1, declared),
            (284, 3, 1, 1), // chunky
            (296, 3, 1, 2),
        ];
        let bps_off = 8 + 2 + entries.len() * 12 + 4;
        let data_off = bps_off + 8;
        let mut f = Vec::with_capacity(data_off + (w as usize) * (h as usize) * 3 * (bits as usize / 8));
        f.extend_from_slice(b"II*\0");
        f.extend_from_slice(&8u32.to_le_bytes());
        f.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, typ, count, value) in entries {
            let value = match tag {
                258 => bps_off as u32,
                273 => data_off as u32,
                _ => value,
            };
            f.extend_from_slice(&tag.to_le_bytes());
            f.extend_from_slice(&typ.to_le_bytes());
            f.extend_from_slice(&count.to_le_bytes());
            f.extend_from_slice(&value.to_le_bytes());
        }
        f.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..3 {
            f.extend_from_slice(&bits.to_le_bytes());
        }
        f.extend_from_slice(&[0, 0]);
        assert_eq!(f.len(), data_off);
        f.resize(data_off + (w as usize) * (h as usize) * 3 * (bits as usize / 8), fill);
        f
    }

    /// Issue #1983: Lightroom and Photoshop write uncompressed TIFFs as one strip, and the `tiff`
    /// crate refuses any strip larger than `intermediate_buffer_size` (128 MiB by default) before
    /// reading it. The bound follows the file for methods that stream; JPEG keeps the default.
    #[test]
    fn a_strip_may_be_as_large_as_the_file_unless_jpeg() {
        let l = limits(&DecodeOptions::default(), 200 << 20, Some(1));
        assert_eq!(l.intermediate_buffer_size, 200 << 20);
        assert_eq!(limits(&DecodeOptions::default(), 200 << 20, None).intermediate_buffer_size, 200 << 20);
        // smaller files keep the old bound: nothing that decoded before is refused
        assert_eq!(limits(&DecodeOptions::default(), 10 << 20, Some(1)).intermediate_buffer_size, 128 << 20);
        // the crate copies a JPEG chunk before decoding it: its declared size stays bounded as before
        assert_eq!(limits(&DecodeOptions::default(), 200 << 20, Some(COMPRESSION_JPEG)).intermediate_buffer_size, 128 << 20);
        assert_eq!(l.decoding_buffer_size, (1usize << 30) * 32);
    }

    /// The reporter's layout: 16-bit RGB 3923 × 5877 as one strip of 131.9 MiB, decoded with the
    /// thumbnail path's options (64 MP). Before the fix: `Malformed(Tiff, "decoder limits exceeded")`.
    #[test]
    fn a_single_strip_over_128_mib_decodes() {
        let (w, h) = (3923u32, 5877u32);
        let strip = w * h * 3 * 2;
        let f = single_strip_rgb(w, h, 16, strip, 0x80);
        assert!(strip > 128 << 20 && f.len() > 128 << 20, "{} bytes", f.len());
        let d = decode(&f, &DecodeOptions { max_size: Some((64, 64)), max_pixels: 64_000_000 }).unwrap();
        assert_eq!(d.bit_depth, 16);
        assert!(d.image.width <= 64 && d.image.height <= 64, "{}×{}", d.image.width, d.image.height);
        // every sample is 0x8080 of 0xffff
        let v = lightcraft_color::transfer::srgb_to_linear(0x8080 as f32 / 65535.0);
        assert!(d.image.data.iter().all(|p| p.iter().all(|c| (c - v).abs() < 1e-3)), "{:?}", d.image.data.first());
    }

    /// A count larger than the floor and the file (200 MiB declared, 48 bytes present) is an error,
    /// not a panic and not an image.
    #[test]
    fn a_strip_count_past_the_file_and_the_floor_is_an_error() {
        let f = single_strip_rgb(4, 4, 8, 200 << 20, 0x80);
        let e = decode(&f, &DecodeOptions::default()).expect_err("refused");
        assert!(matches!(e, Error::Malformed(..)), "{e:?}");
    }
}
